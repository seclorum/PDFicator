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
    faiss_index INTEGER  -- Store FAISS index for each document
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
            faiss_index_id = doc_id  # Map document ID to FAISS index
            documents.append((file, content, faiss_index_id))  # Add document with its ID for FAISS mapping

# Commit changes
conn.commit()

# Vectorize content and build FAISS index
texts = [doc[1] for doc in documents]
embeddings = model.encode(texts, convert_to_tensor=True).cpu().numpy()
d = embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(d)

# Ensure that the document IDs are correctly added to FAISS
for i, text in enumerate(texts):
    print(f"Inserting document {i} into FAISS")  # Debugging print
faiss_index.add(embeddings)

# Save the FAISS index to file
faiss.write_index(faiss_index, 'data/faiss_index.bin')

print("Indexing complete.")

# Close the database connection
conn.close()

