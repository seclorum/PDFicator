import os
import sqlite3
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

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
    faiss_index INTEGER  -- Store FAISS index position
)
''')

cursor.execute('''
CREATE VIRTUAL TABLE IF NOT EXISTS document_index USING FTS5(
    content
)
''')

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
faiss_index = faiss.IndexFlatL2(384)  # Set FAISS index dimension (384 for MiniLM-L6-v2)

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
            doc_id = cursor.lastrowid  # Get the last inserted document ID

            cursor.execute('INSERT INTO document_index (content) VALUES (?)', (content,))
            documents.append((doc_id, content))  # Add document ID and content for FAISS mapping

# Commit changes
conn.commit()

# Vectorize content and build FAISS index
texts = [doc[1] for doc in documents]
embeddings = model.encode(texts, convert_to_tensor=True).cpu().numpy()

# Add embeddings to FAISS and update database with FAISS index positions
for i, (doc_id, _) in enumerate(documents):
    embedding = embeddings[i:i+1]  # FAISS expects 2D arrays
    faiss_index.add(embedding)
    # Update SQLite with the FAISS index position
    cursor.execute('UPDATE documents SET faiss_index = ? WHERE id = ?', (i, doc_id))

# Save the FAISS index to file
faiss.write_index(faiss_index, 'data/faiss_index.bin')

print("Indexing complete.")

# Close the database connection
conn.close()

