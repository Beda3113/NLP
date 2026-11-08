"""Точка входа CLI для Document Vectorizer."""
import os
import signal
import warnings
import multiprocessing as mp
from pathlib import Path
from functools import wraps

import click
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# Подавление шумных предупреждений библиотек
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
warnings.filterwarnings("ignore", category=UserWarning, module="pdfminer")
warnings.filterwarnings("ignore", category=UserWarning, module="pdfplumber")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="sentence_transformers")

from src.utils.logger import setup_logger
from src.utils.config import load_config
from src.utils.file_utils import scan_files
from src.processors.registry import ProcessorRegistry
from src.vectorization.encoder import GPUEncoder
from src.vectorization.chunker import TextChunker
from src.storage.vector_db import VectorDB
from src.search.searcher import Searcher


# ============================================================================
# Механизм таймаута для параллельных процессов
# ============================================================================

class FileTimeoutError(Exception):
    """Исключение для прерывания обработки зависшего файла."""
    pass

def timeout_handler(signum, frame):
    raise FileTimeoutError("Превышено время обработки файла (90 сек)")

def with_timeout(seconds):
    """Декоратор для установки жесткого таймаута на выполнение функции."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # SIGALRM работает только в главном потоке процесса, но в multiprocessing 
            # каждый процесс имеет свой главный поток, так что это безопасно.
            if os.name != 'nt':
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                if os.name != 'nt':
                    signal.alarm(0)
            return result
        return wrapper
    return decorator

# ============================================================================
# Вспомогательные функции для multiprocessing
# ============================================================================

@with_timeout(90)  # ⏱️ УВЕЛИЧЕНО ДО 90 СЕКУНД для тяжелых PDF
def _process_single_file(args):
    """
    Функция обработки одного файла в отдельном процессе.
    """
    file_path, config = args
    try:
        registry = ProcessorRegistry(config)
        text = registry.process(Path(file_path))
        
        if not text or len(text.strip()) < config["processing"]["chunking"]["min_chunk_size"]:
            return None
        
        chunker = TextChunker(config)
        chunks = chunker.chunk(text)
        
        if not chunks:
            return None
        
        result_chunks = []
        for i, chunk in enumerate(chunks):
            result_chunks.append({
                "text": chunk,
                "source": str(file_path),
                "chunk_id": i,
                "total_chunks": len(chunks),
            })
        return result_chunks
    except FileTimeoutError:
        # Тихо пропускаем файлы, которые не успели обработаться
        return None
    except Exception:
        # Тихо пропускаем битые файлы
        return None

# ============================================================================
# CLI команды
# ============================================================================

@click.group()
def cli():
    """Document Vectorizer — конвертация документов в векторную базу."""
    pass

@cli.command()
@click.option("--input", "input_dir", required=True, help="Папка с документами")
@click.option("--output", "output_dir", default="./vector_db", help="Папка для базы")
@click.option("--config", "config_path", default="config/default.yaml", help="Конфиг")
@click.option("--batch-size", default=None, type=int, help="Размер батча для GPU")
@click.option("--workers", default=None, type=int,
              help="Количество CPU-потоков для чтения (по умолчанию: половина ядер)")
def index(input_dir, output_dir, config_path, batch_size, workers):
    """Индексация папки с документами."""
    logger = setup_logger()
    cfg = load_config(config_path)
    
    if not cfg:
        cfg = {
            "processing": {
                "supported_formats": "xlsx,xls,csv,pdf,jpg,jpeg,png,tiff,bmp,docx,txt,md",
                "ignored_formats": "mp4,mov,mpg,mpeg,avi,mkv,wmv,flv,webm,mp3,wav,flac,aac,ogg,wma,m4a,exe,msi,zip,rar,7z,tar,gz",
                "ocr": {"languages": "rus+eng", "psm": 6},
                "chunking": {"chunk_size": 512, "overlap": 50, "min_chunk_size": 20}
            },
            "vectorization": {
                "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "batch_size": 64,
                "normalize": True
            },
            "gpu": {"use_multi_gpu": True, "devices": [0, 1]},
            "storage": {"backend": "faiss", "index_type": "IndexFlatIP"}
        }
        
    if batch_size:
        cfg["vectorization"]["batch_size"] = batch_size

    input_path = Path(input_dir).resolve()
    output_path = Path(output_dir).resolve()
    
    if not input_path.exists():
        logger.error(f"Папка не найдена: {input_path}")
        return

    # Определяем количество рабочих потоков
    if workers is None:
        workers = max(1, mp.cpu_count() // 2)
    workers = min(workers, mp.cpu_count())

    logger.info(f"Сканирование: {input_path}")
    
    files = scan_files(
        input_path,
        cfg["processing"]["supported_formats"],
        cfg["processing"].get("ignored_formats")
    )
    
    logger.info(f"Найдено файлов: {len(files)}")
    if not files:
        logger.warning("Нет файлов для обработки")
        return

    logger.info("Инициализация GPU-энкодера...")
    encoder = GPUEncoder(cfg)
    db = VectorDB(output_path, cfg, dimension=encoder.dimension)
    db.load_or_create()

    # Фильтрация уже обработанных файлов
    files_to_process = [f for f in files if not db.is_processed(f)]
    logger.info(f"Файлов для обработки: {len(files_to_process)} (пропущено: {len(files) - len(files_to_process)})")
    
    if not files_to_process:
        logger.info("Все файлы уже обработаны. База актуальна.")
        encoder.close()
        return

    logger.info(f"Запуск параллельного чтения на {workers} потоках CPU...")
    
    tasks = [(str(f), cfg) for f in files_to_process]
    all_chunks = []
    processed_files = []
    failed_count = 0
    
    # Используем spawn для безопасности (избегаем проблем с fork + CUDA)
    ctx = mp.get_context('spawn')
    
    with ctx.Pool(processes=workers) as pool:
        for result in tqdm(
            pool.imap_unordered(_process_single_file, tasks, chunksize=10),
            total=len(tasks),
            desc="Чтение файлов (CPU)",
            unit="file"
        ):
            if result:
                all_chunks.extend(result)
                processed_files.append(result[0]["source"])
            else:
                failed_count += 1

    # Помечаем успешно обработанные файлы
    for file_path_str in processed_files:
        db.mark_processed(Path(file_path_str))
        
    logger.info(f"✅ Прочитано: {len(processed_files)} | Пропущено/ошибки/таймауты: {failed_count} | Чанков: {len(all_chunks)}")
    
    if not all_chunks:
        logger.warning("Нет чанков для векторизации")
        encoder.close()
        return

    # ========================================================================
    # ЭТАП 2: Векторизация на GPU
    # ========================================================================
    logger.info(f"Векторизация {len(all_chunks)} чанков на GPU...")
    texts = [c["text"] for c in all_chunks]
    
    embeddings = encoder.encode(texts)

    # ========================================================================
    # ЭТАП 3: Сохранение в FAISS
    # ========================================================================
    logger.info("Сохранение векторов в базу...")
    db.add(embeddings, all_chunks)
    db.save()
    
    # Корректное завершение multi-GPU пула
    encoder.close()
    
    logger.info(f"🎉 Готово! Векторов в базе: {db.count()}")
    logger.info(f"📁 База сохранена в: {output_path}")

@cli.command()
@click.option("--db", "db_path", required=True, help="Папка с векторной базой")
@click.option("--query", required=True, help="Поисковый запрос")
@click.option("--top-k", default=10, type=int, help="Количество результатов")
@click.option("--output", default=None, help="Сохранить результаты в файл (JSON)")
@click.option("--config", "config_path", default="config/default.yaml", help="Конфиг")
def search(db_path, query, top_k, output, config_path):
    """Поиск по векторной базе."""
    import json
    logger = setup_logger()
    cfg = load_config(config_path)
    
    if not cfg:
        cfg = {
            "vectorization": {
                "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "batch_size": 32,
                "normalize": True
            },
            "gpu": {"use_multi_gpu": False, "devices": [0]}
        }
        
    # ИСПРАВЛЕНО: for_query=True для корректного применения query-префикса
    encoder = GPUEncoder(cfg, for_query=True)
    db = VectorDB(Path(db_path), cfg)
    db.load()
    
    if db.count() == 0:
        logger.error("База данных пуста")
        encoder.close()
        return
        
    searcher = Searcher(db, encoder)
    results = searcher.search(query, top_k=top_k)
    
    # Если указан --output, сохраняем в JSON
    if output:
        with open(output, "w", encoding="utf-8") as f:
            json.dump({"query": query, "results": results}, f, ensure_ascii=False, indent=2)
        logger.info(f"Результаты сохранены в: {output}")
        encoder.close()
        return
    
    # Иначе выводим в консоль
    click.echo(f"\n🔍 Результаты для: '{query}'\n")
    click.echo(f"{'=' * 80}")
    for i, r in enumerate(results, 1):
        click.echo(f"\n[{i}] Score: {r['score']:.4f}")
        click.echo(f"    Source: {r['source']}")
        click.echo(f"    Chunk:  {r['chunk_id'] + 1}/{r['total_chunks']}")
        click.echo(f"    Text:   {r['text'][:300]}...")
        click.echo(f"{'-' * 80}")
        
    click.echo(f"\nНайдено результатов: {len(results)}")
    encoder.close()

@cli.command()
@click.option("--db", "db_path", required=True, help="Папка с векторной базой")
def info(db_path):
    """Информация о векторной базе."""
    db = VectorDB(Path(db_path), {})
    db.load()
    
    click.echo("\n📊 Информация о векторной базе")
    click.echo(f"{'=' * 50}")
    click.echo(f"Путь:              {db.path}")
    click.echo(f"Векторов:          {db.count()}")
    click.echo(f"Обработано файлов: {db.processed_count()}")
    click.echo(f"Размерность:       {db.dimension}")
    click.echo(f"{'=' * 50}\n")

@cli.command()
@click.option("--db", "db_path", required=True, help="Папка с векторной базой")
@click.option("--query", required=True, help="Поисковый запрос")
@click.option("--top-k", default=5, type=int, help="Количество результатов")
@click.option("--output", default=None, help="Сохранить результаты в файл (JSON)")
@click.option("--config", "config_path", default="config/default.yaml", help="Конфиг")
def export(db_path, query, top_k, output, config_path):
    """Экспорт результатов поиска в JSON."""
    import json
    logger = setup_logger()
    cfg = load_config(config_path)
    
    if not cfg:
        cfg = {
            "vectorization": {
                "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "batch_size": 32,
                "normalize": True
            },
            "gpu": {"use_multi_gpu": False, "devices": [0]}
        }
        
    # ИСПРАВЛЕНО: for_query=True для корректного применения query-префикса
    encoder = GPUEncoder(cfg, for_query=True)
    db = VectorDB(Path(db_path), cfg)
    db.load()
    
    searcher = Searcher(db, encoder)
    results = searcher.search(query, top_k=top_k)
    
    if output:
        with open(output, "w", encoding="utf-8") as f:
            json.dump({"query": query, "results": results}, f, ensure_ascii=False, indent=2)
        logger.info(f"Результаты сохранены в: {output}")
    else:
        click.echo(json.dumps(results, ensure_ascii=False, indent=2))
        
    encoder.close()

if __name__ == "__main__":
    cli()
