import faiss
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
import hashlib

# Paths to FAISS index and SQLite database
faiss_index_path = 'data/faiss_index.bin'
db_path = 'data/pdf_index.db'

# Load FAISS index
faiss_index = faiss.read_index(faiss_index_path)
num_vectors = faiss_index.ntotal
dimension = faiss_index.d

# Hashing function to convert FAISS index to a unique hash
def hash_index(idx):
    """Hash the FAISS index to compare with stored hashes."""
    return hashlib.sha256(str(idx).encode('utf-8')).hexdigest()

print(f"FAISS Index Loaded:")
print(f"- Number of Vectors: {num_vectors}")
print(f"- Dimension of Each Vector: {dimension}")

# Retrieve all vectors
vectors = np.zeros((num_vectors, dimension), dtype=np.float32)
faiss_index.reconstruct_n(0, num_vectors, vectors)

# Connect to SQLite
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute("SELECT id, filename, content, keywords, faiss_index FROM documents")
db_records = cursor.fetchall()

# Compare FAISS index and SQLite records
print("\nComparing FAISS Index with SQLite Records:")
mismatches = 0
for record in db_records:
    doc_id, filename, content, keywords, db_faiss_index = record

    # Handle missing indices
    if db_faiss_index is None:
        print(f"Error: FAISS index is NULL for document '{filename}' (ID: {doc_id})")
        mismatches += 1
        continue

    # Hash the FAISS index stored in SQLite to compare with FAISS index
    hashed_db_faiss_index = db_faiss_index.strip()

    # Validate the FAISS index by searching for the matching hash
    try:
        # Loop through the FAISS index to find a match
        match_found = False
        for i in range(num_vectors):
            faiss_index_hash = hash_index(i)
            if faiss_index_hash == hashed_db_faiss_index:
                faiss_vector = vectors[i]
                print(f"Document '{filename}' (SQLite ID: {doc_id}, FAISS Index: {faiss_index_hash})")
                print(f"- Keywords: {keywords[:100]}")
                print(f"- FAISS Vector Sample: {faiss_vector[:5]}")  # First 5 dimensions
                match_found = True
                break

        if not match_found:
            print(f"Error: No match found for FAISS index {hashed_db_faiss_index} for document '{filename}' (ID: {doc_id})")
            mismatches += 1

    except IndexError:
        print(f"Error: Invalid FAISS index for document '{filename}' (ID: {doc_id}): {hashed_db_faiss_index}")
        mismatches += 1

# Optional: Re-encode content and validate (first 5 records)
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
print("\nValidating embeddings by re-encoding content...")
for record in db_records[:5]:  # Limit validation for efficiency
    doc_id, filename, content, keywords, db_faiss_index = record
    if db_faiss_index is not None:
        # Re-encode document content
        reencoded_vector = model.encode([content], convert_to_tensor=True).cpu().numpy()[0]

        # Find the corresponding FAISS index by matching the hash
        hashed_db_faiss_index = db_faiss_index.strip()
        match_found = False
        for i in range(num_vectors):
            faiss_index_hash = hash_index(i)
            if faiss_index_hash == hashed_db_faiss_index:
                faiss_vector = vectors[i]
                match_found = True
                break

        if match_found:
            # Compute difference
            diff = np.linalg.norm(reencoded_vector - faiss_vector)
            print(f"Document '{filename}' (ID: {doc_id}) - Embedding Difference: {diff:.6f}")
        else:
            print(f"Error: No match found for FAISS index {hashed_db_faiss_index} for document '{filename}' (ID: {doc_id})")
            mismatches += 1

# Summary
if mismatches > 0:
    print(f"\nTotal mismatches found: {mismatches}")
else:
    print("\nNo mismatches found. FAISS and SQLite are synchronized.")

conn.close()

