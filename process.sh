#!/bin/zsh

# --- Stage 2: Analysis and Indexing ---

# This should already have been set up by tools.sh
source pdf_tool_env/bin/activate

# Define directories
PDF_DIR="archive/testCollection"
DB_PATH="data/pdf_index.db"

# Run the processing script
python py/process_pdfs.py


deactivate
