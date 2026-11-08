from setuptools import setup, find_packages

setup(
    name="document-vectorizer",
    version="1.0.0",
    description="Конвертер документов в векторную базу с GPU",
    packages=find_packages(),
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "docvec=src.main:cli",
        ],
    },
)
