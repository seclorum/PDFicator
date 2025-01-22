import sqlite3
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Configuration
db_path = 'data/pdf_index.db'
faiss_index_path = 'faiss_index.bin'

# Load database and FAISS index
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

faiss_index = faiss.read_index(faiss_index_path)
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Sample query
query = "Machine learning applications in biology"
print(f"Query: {query}")
query_embedding = model.encode([query], convert_to_tensor=True).cpu().numpy()

# Search FAISS index
D, I = faiss_index.search(query_embedding, k=5)

# Fetch results
for idx in I[0]:
    if idx == -1:
        continue
    cursor.execute('SELECT filename, keywords FROM documents WHERE id = ?', (idx + 1,))
    result = cursor.fetchone()
    print(f"File: {result[0]}, Keywords: {result[1]}")

conn.close()
