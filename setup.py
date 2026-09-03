from setuptools import setup, find_packages

setup(
    name="hyper-ocr",
    version="3.0.0",
    description="Ultra-Fast Hardware-Accelerated Real-Time OCR & Screen Text Extractor",
    author="JTG Systems",
    author_email="jtgsystems@gmail.com",
    url="https://jtgsystems.com",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "hyper-ocr=hyper_ocr.cli:main",
            "hocr=hyper_ocr.cli:main",
        ],
    },
)
