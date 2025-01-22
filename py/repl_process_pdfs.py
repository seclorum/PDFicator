import os
import sqlite3
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import readline
from sklearn.feature_extraction.text import TfidfVectorizer
import time

# Configuration
pdf_dir = os.path.expanduser('archive/testCollection')
db_path = 'data/pdf_index.db'
faiss_index_path = 'data/faiss_index.bin'

# Function to safely close the database
def safe_close_db(conn):
    try:
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error closing database: {e}")

# Initialize database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    filename TEXT,
    content TEXT,
    keywords TEXT
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

# Function to load FAISS index or create a new one if not exists
def load_or_create_faiss_index():
    if os.path.exists(faiss_index_path):
        print("Loading existing FAISS index...")
        return faiss.read_index(faiss_index_path)
    else:
        print("Creating new FAISS index...")
        # Initialize the FAISS index with the expected dimensionality
        return faiss.IndexFlatL2(768)  # Assuming 768 dimensions for the SentenceTransformer model

# Function to process PDFs incrementally
def process_pdfs(faiss_index):
    documents = []  # List to store documents as they are processed
    stop_processing = False

    for root, _, files in os.walk(pdf_dir):
        if stop_processing:
            break  # Exit the loop if interrupted
        
        for file in files:
            if file.endswith('.pdf'):
                filepath = os.path.join(root, file)
                print(f"Processing: {filepath}")
                content = extract_text_from_pdf(filepath)
                keywords = ", ".join(content.split()[:20])  # Placeholder keyword extraction

                # Save to database
                cursor.execute('INSERT INTO documents (filename, content, keywords) VALUES (?, ?, ?)',
                               (file, content, keywords))

                cursor.execute('INSERT INTO document_index (content) VALUES (?)', (content,))
                documents.append((file, content))

                # Commit after each file processed (incremental)
                conn.commit()

                # Incrementally build FAISS index
                texts = [doc[1] for doc in documents]
                embeddings = model.encode(texts, convert_to_tensor=True).cpu().numpy()

                # Check that the embeddings dimensionality is consistent (768)
                assert embeddings.shape[1] == 768, "Embedding dimensionality mismatch!"

                faiss_index.add(embeddings)  # Add the new embeddings to the FAISS index

                # Save the FAISS index to disk incrementally
                faiss.write_index(faiss_index, faiss_index_path)

                # Prompt for user input to allow interruption
                print("\nPress Enter to pause processing, or type 'continue' to keep going.")
                user_input = input("Your command: ").strip().lower()

                if user_input == "continue":
                    continue  # Continue processing PDFs
                elif user_input == "":  # Pressing Enter with no input pauses the process
                    print("\nProcessing paused. Type 'continue' to resume.")
                    return stop_processing  # Return to allow interruption/continuation
                elif user_input == "end":
                    print("Ending the process.")
                    stop_processing = True
                    return stop_processing  # Stop the process

    return stop_processing

# Query function to search in FAISS index
def query_faiss_index(faiss_index):
    query = input("Enter query: ").strip()
    if query:
        query_embedding = model.encode([query], convert_to_tensor=True).cpu().numpy()
        results = faiss_index.search(query_embedding, k=5)

        for idx, dist in zip(results[0][0], results[1][0]):
            if idx != -1:  # If idx is -1, it means no valid match was found
                int_idx = int(round(idx))  # Convert FAISS float index to integer

                cursor.execute('SELECT filename, keywords FROM documents WHERE id = ?', (int_idx + 1,))
                result = cursor.fetchone()
                if result:
                    file_name, keywords = result
                    print(f"File: {file_name}, Keywords: {keywords[:100]}")  # Adjust for the content display
                else:
                    print(f"No document found for index {int_idx}")
            else:
                print("No valid results found for the query.")
    else:
        print("Query was empty. Please enter a valid query.")

# Start processing PDFs with REPL-like interface
def run_repl():
    faiss_index = load_or_create_faiss_index()  # Load or create FAISS index
    while True:
        print("\nType 'continue' to start or resume processing, 'query' to search the data, or 'end' to finish.")
        command = input("REPL command: ").strip().lower()

        if command == "continue":
            stop_processing = process_pdfs(faiss_index)
            if stop_processing:
                break  # Exit the loop if processing is stopped

        elif command == "query":
            query_faiss_index(faiss_index)

        elif command == "end":
            print("Ending the process and safely closing the database.")
            safe_close_db(conn)
            break

        else:
            print("Invalid command. Please enter 'continue', 'query', or 'end'.")

# Start the REPL
try:
    run_repl()
except KeyboardInterrupt:
    print("\nProcess interrupted. Closing the database safely.")
    safe_close_db(conn)

