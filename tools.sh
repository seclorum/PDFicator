#!/bin/zsh

# --- Stage 1: Installation and Setup ---
# Notes: MacOS/Linux versions of commands are given
# 
echo "Setting up dependencies..."

# Update package manager and install prerequisites
sudo apt-get update -y || brew update
sudo apt-get install -y python3 python3-pip sqlite3 poppler-utils || brew install python3 sqlite3 poppler

# Create and activate a virtual environment
python3 -m venv pdf_tool_env
source pdf_tool_env/bin/activate

# Install required Python libraries
pip install --upgrade pip
pip install PyPDF2 pdfminer.six pytesseract transformers whoosh scikit-learn
pip install sentence-transformers faiss-cpu


