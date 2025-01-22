import sqlite3
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import readline  # Required for interactive input in the terminal

# Configuration
db_path = 'data/pdf_index.db'
faiss_index_path = 'data/faiss_index.bin'

# Load database and FAISS index
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

faiss_index = faiss.read_index(faiss_index_path)
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def print_top_subjects():
    # Querying with a dummy query (could be empty or anything, as we only want to get top matches)
    dummy_query = "sample"
    query_embedding = model.encode([dummy_query], convert_to_tensor=True).cpu().numpy()

    # Run the FAISS search for top 10 closest documents
    results = faiss_index.search(query_embedding, k=10)  # Retrieve top 10 matches

    print("Top 10 subjects in the FAISS index:")
    # Assuming results[0] are the indices and results[1] are the distances
    for idx, dist in zip(results[0][0], results[1][0]):  # Indexing results arrays
        if idx != -1:
            # Adjust FAISS index to SQLite index (FAISS is 0-based, SQLite is 1-based)
            int_idx = int(round(idx)) + 1  # Convert FAISS float index to integer and shift to SQLite's 1-based index
            
            # Check if the index is within a valid range in the SQLite database
            cursor.execute('SELECT filename, keywords FROM documents WHERE faiss_index = ?', (idx,))
            result = cursor.fetchone()
            if result:
                file_name, keywords = result
                print(f"File: {file_name}, Keywords: {keywords[:100]}")  # Display top 100 characters of keywords
            else:
                print(f"No document found for index {int_idx}")
        else:
            print("No valid results found for the query.")

def search(query):
    print(f"Query: {query}")
    query_embedding = model.encode([query], convert_to_tensor=True).cpu().numpy()

    # Run the FAISS search
    results = faiss_index.search(query_embedding, k=5)  # or whatever number of results you need
    
    # Check if results are empty or invalid
    if results[0].size == 0 or results[1].size == 0:
        print("No valid results found for the query.")
        return

    # Assuming results[0] are the indices and results[1] are the distances
    for idx, dist in zip(results[0][0], results[1][0]):  # Note results[0] and results[1] are arrays, indexing them
        if idx != -1:  # If idx is -1, it means no valid match was found
            int_idx = int(round(idx)) + 1  # Convert FAISS float index to integer and adjust for SQLite's 1-based index

            # Adjust indexing if FAISS and database IDs don't align
            cursor.execute('SELECT filename, keywords FROM documents WHERE faiss_index = ?', (idx,))
            result = cursor.fetchone()
            if result:
                file_name, keywords = result
                print(f"File: {file_name}, Keywords: {keywords[:100]}")  # Adjust for the content display
            else:
                print(f"No document found for index {int_idx}")
        else:
            print("No valid results found for the query.")

# REPL loop
print("Welcome to the PDF search REPL. Type your query and press Enter.")
print("Type 'exit' to quit.")
print("Type 'top' to see the top 10 subjects in the FAISS index.")

while True:
    try:
        # Read input from the user
        query = input("Query: ").strip()

        if query.lower() == 'exit':
            print("Exiting...")
            break
        elif query.lower() == 'top':
            print_top_subjects()  # Print the top 10 subjects
        else:
            # Call the search function with the user's query
            search(query)
        
    except KeyboardInterrupt:
        print("\nExiting...")
        break
    except Exception as e:
        print(f"Error: {e}")

# Close the database connection
conn.close()

