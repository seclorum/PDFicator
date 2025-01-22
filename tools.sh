#!/bin/zsh

# --- Stage 1: Installation and Setup ---
# Notes: Updated for Python 3.11+ compatibility
echo "Making local directory structure..."
mkdir -p archive/testCollection data

echo "Setting up dependencies..."

# Update package manager and install prerequisites
# !J! Only do these if you don't already have your python eggs in order
#sudo apt-get update -y || brew update
#sudo apt-get install -y python3.11 python3.11-venv python3.11-dev sqlite3 poppler-utils || brew install python3 sqlite3 poppler
 
# Create and activate a virtual environment
python3.11 -m venv pdf_tool_env
source pdf_tool_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Pin NumPy to a version below 2.0
pip install "numpy<2"
pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/cpu
pip install sentence-transformers==2.2.2 huggingface_hub==0.4.0
pip install PyPDF2 pdfminer.six pytesseract transformers whoosh scikit-learn faiss-cpu

echo "Dependencies successfully set up!"

