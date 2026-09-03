#!/usr/bin/env bash
# JTG Systems - HyperOCR Launcher (Linux / macOS)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

if [ $# -eq 0 ]; then
    python3 -m hyper_ocr.cli --help
else
    python3 -m hyper_ocr.cli "$@"
fi
