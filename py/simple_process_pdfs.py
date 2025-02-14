import os
import sqlite3
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import hashlib

# Configuration
pdf_dir = os.path.expanduser('archive/testCollection')
db_path = 'data/pdf_index.db'

# Initialize database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    filename TEXT,
    content TEXT,
    keywords TEXT,
    faiss_index TEXT  -- Changed to TEXT to store hash values
)
''')

cursor.execute('''
CREATE VIRTUAL TABLE IF NOT EXISTS document_index USING FTS5(
    content
)
''')

# Hashing function to convert FAISS index to a unique hash
def hash_index(idx):
    return hashlib.sha256(str(idx).encode('utf-8')).hexdigest()

# Text extraction function
def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        return " ".join([page.extract_text() for page in reader.pages])
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

# Initialize NLP model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Process each PDF
documents = []
for root, _, files in os.walk(pdf_dir):
    for file in files:
        if file.endswith('.pdf'):
            filepath = os.path.join(root, file)
            print(f"Processing: {filepath}")
            content = extract_text_from_pdf(filepath)
            keywords = ", ".join(content.split()[:20])  # Placeholder keyword extraction

            # Save to database
            cursor.execute('INSERT INTO documents (filename, content, keywords) VALUES (?, ?, ?)',
                           (file, content, keywords))
            doc_id = cursor.lastrowid
            cursor.execute('INSERT INTO document_index (content) VALUES (?)', (content,))
            documents.append((doc_id, content))

# Commit changes to save initial data
conn.commit()

# Vectorize content and build FAISS index
texts = [doc[1] for doc in documents]
embeddings = model.encode(texts, convert_to_tensor=True).cpu().numpy()
d = embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(d)

# Add vectors to FAISS and update SQLite with hashed FAISS indices
faiss_index.add(embeddings)
for i, (doc_id, _) in enumerate(documents):
    # Generate hash for the FAISS index
    faiss_index_hash = hash_index(i)
    cursor.execute('UPDATE documents SET faiss_index = ? WHERE id = ?', (faiss_index_hash, doc_id))
conn.commit()

# Save the FAISS index to file
faiss.write_index(faiss_index, 'data/faiss_index.bin')

print("Indexing complete.")
conn.close()

