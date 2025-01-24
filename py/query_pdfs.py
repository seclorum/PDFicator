import sqlite3
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Configuration
db_path = 'data/pdf_index.db'
faiss_index_path = 'data/faiss_index.bin'

# Load database and FAISS index
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
faiss_index = faiss.read_index(faiss_index_path)
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def print_top_subjects():
    query_embedding = model.encode(["sample"], convert_to_tensor=True).cpu().numpy()

    # Run the FAISS search for top 10 closest documents
    results = faiss_index.search(query_embedding, k=10)

    print("Top 10 subjects in the FAISS index:")
    for idx, dist in zip(results[0][0], results[1][0]):
        if idx != -1:
            cursor.execute('SELECT filename, keywords FROM documents WHERE faiss_index = ?', (int(idx),))
            result = cursor.fetchone()
            if result:
                file_name, keywords = result
                print(f"File: {file_name}, Keywords: {keywords[:100]}")  # Top 100 characters of keywords
            else:
                print(f"No document found for FAISS index {idx}")
        else:
            print("No valid results found.")

def search(query):
    print(f"Query: {query}")
    query_embedding = model.encode([query], convert_to_tensor=True).cpu().numpy()

    # Run the FAISS search
    results = faiss_index.search(query_embedding, k=5)
    for idx, dist in zip(results[0][0], results[1][0]):
        if idx != -1:
            cursor.execute('SELECT filename, keywords FROM documents WHERE faiss_index = ?', (int(idx),))
            result = cursor.fetchone()
            if result:
                file_name, keywords = result
                print(f"File: {file_name}, Keywords: {keywords[:100]}")
            else:
                print(f"No document found for FAISS index {idx}")
        else:
            print("No valid results found.")

# REPL loop
print("Welcome to the PDF search REPL. Type your query and press Enter.")
print("Type 'exit' to quit.")
print("Type 'top' to see the top 10 subjects in the FAISS index.")

while True:
    try:
        query = input("Query: ").strip()
        if query.lower() == 'exit':
            print("Exiting...")
            break
        elif query.lower() == 'top':
            print_top_subjects()
        else:
            search(query)
    except KeyboardInterrupt:
        print("\nExiting...")
        break
    except Exception as e:
        print(f"Error: {e}")

conn.close()

