#!/bin/zsh

# --- Stage 1: Installation and Setup ---
# Notes: Updated for Python 3.11+ compatibility
echo "Setting up dependencies..."

# Update package manager and install prerequisites
# !J! Only do these if you don't already have your python eggs in order
#sudo apt-get update -y || brew update
#sudo apt-get install -y python3.11 python3.11-venv python3.11-dev sqlite3 poppler-utils || brew install python3 sqlite3 poppler

# Create and activate a virtual environment using Python 3.11
python3.11 -m venv pdf_tool_env
source pdf_tool_env/bin/activate

# Upgrade pip to the latest version
pip install --upgrade pip

# Install specific compatible versions of sentence-transformers and torch for Python 3.11+
pip install torch==2.0.0+cpu torchvision==0.15.1+cpu torchaudio==2.0.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install sentence-transformers==2.2.2

# Install other required Python libraries
pip install PyPDF2 pdfminer.six pytesseract transformers whoosh scikit-learn faiss-cpu

echo "Dependencies successfully set up!"

