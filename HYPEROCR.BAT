@echo off
setlocal
title JTG Systems - HyperOCR Real-Time Text Extractor

cd /d "%~dp0"
set PYTHONPATH=%CD%;%PYTHONPATH%

if "%~1"=="" (
    python -m hyper_ocr.cli --help
) else (
    python -m hyper_ocr.cli %*
)
pause
