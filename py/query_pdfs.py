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

def search(query):
    print(f"Query: {query}")
    query_embedding = model.encode([query], convert_to_tensor=True).cpu().numpy()

    # Run the FAISS search
    results = faiss_index.search(query_embedding, k=5)  # or whatever number of results you need

    # Assuming results[0] are the indices and results[1] are the distances
    for idx, dist in zip(results[0][0], results[1][0]):  # Note results[0] and results[1] are arrays, indexing them
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

# REPL loop
print("Welcome to the PDF search REPL. Type your query and press Enter.")
print("Type 'exit' to quit.")

while True:
    try:
        # Read input from the user
        query = input("Query: ").strip()

        if query.lower() == 'exit':
            print("Exiting...")
            break

        # Call the search function with the user's query
        search(query)
        
    except KeyboardInterrupt:
        print("\nExiting...")
        break
    except Exception as e:
        print(f"Error: {e}")

# Close the database connection
conn.close()

