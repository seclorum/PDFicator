import os
import sqlite3
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import readline
from sklearn.feature_extraction.text import TfidfVectorizer

# Configuration
pdf_dir = os.path.expanduser('archive/testCollection')
db_path = 'data/pdf_index.db'

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

# REPL to interact with the process
def run_repl():
    processing = False  # Flag to track whether we are currently processing
    documents = []  # List to store documents as they are processed

    while True:
        # Instead of a regular command input, check for an empty line (Enter key press)
        command = input("REPL command (continue, end, query, or help): ").strip().lower()

        if command == "continue":
            if processing:
                print("Already processing.")
            else:
                print("Starting PDF processing...")
                processing = True
                try:
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

                                cursor.execute('INSERT INTO document_index (content) VALUES (?)', (content,))
                                documents.append((file, content))

                    # Commit changes after processing
                    conn.commit()

                    # Vectorize content and build FAISS index
                    texts = [doc[1] for doc in documents]
                    embeddings = model.encode(texts, convert_to_tensor=True).cpu().numpy()
                    d = embeddings.shape[1]
                    faiss_index = faiss.IndexFlatL2(d)
                    faiss_index.add(embeddings)
                    faiss.write_index(faiss_index, 'data/faiss_index.bin')

                    print("Indexing complete.")
                except KeyboardInterrupt:
                    print("\nProcessing interrupted. Database is safe.")
                    safe_close_db(conn)
                    break

        elif command == "end":
            print("Ending the process and closing the database.")
            safe_close_db(conn)
            break

        elif command == "query":
            query = input("Enter query: ").strip()
            query_embedding = model.encode([query], convert_to_tensor=True).cpu().numpy()
            faiss_index = faiss.read_index('data/faiss_index.bin')
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

        elif command == "help":
            print("Available commands:")
            print("  continue - Continue processing PDFs and indexing.")
            print("  end - End the process and safely close the database.")
            print("  query - Query the indexed documents.")
            print("  help - Show this help message.")

        elif command == "":  # If Enter is pressed with no input, we treat it as an interruption
            print("Enter key pressed. Stopping the process.")
            safe_close_db(conn)
            break

        else:
            print("Invalid command. Type 'help' for a list of commands.")

# Start the REPL
try:
    run_repl()
except KeyboardInterrupt:
    print("\nProcess interrupted. Closing the database safely.")
    safe_close_db(conn)

