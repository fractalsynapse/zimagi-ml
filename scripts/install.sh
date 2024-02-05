#!/usr/bin/env bash
#
# Install module related dependencies
#
set -e

echo "Installing Tesseract OCR"
apt-get update
apt-get install -y tesseract-ocr poppler-utils zlib1g-dev

echo "Installing Pytorch and Spacy"
python3 -m pip install "torch>=2.1.0"
python3 -m pip install "spacy>=3.5.0"
python3 -m spacy download en_core_web_lg
