import faiss
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer

# Paths to FAISS index and SQLite database
faiss_index_path = 'data/faiss_index.bin'
db_path = 'data/pdf_index.db'

# Load FAISS index
faiss_index = faiss.read_index(faiss_index_path)
num_vectors = faiss_index.ntotal
dimension = faiss_index.d

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

    # Convert the float index to integer for FAISS
    db_faiss_index = int(round(db_faiss_index))  # Round and cast to integer

    # Check bounds for FAISS index
    if db_faiss_index < 0 or db_faiss_index >= num_vectors:
        print(f"Error: FAISS index for document '{filename}' (ID: {doc_id}) is out of bounds: {db_faiss_index}")
        mismatches += 1
        continue

    # Validate embeddings
    try:
        faiss_vector = vectors[db_faiss_index]
        print(f"Document '{filename}' (SQLite ID: {doc_id}, FAISS Index: {db_faiss_index})")
        print(f"- Keywords: {keywords[:100]}")
        print(f"- FAISS Vector Sample: {faiss_vector[:5]}")  # First 5 dimensions
    except IndexError:
        print(f"Error: Invalid FAISS index for document '{filename}' (ID: {doc_id}): {db_faiss_index}")
        mismatches += 1

# Optional: Re-encode content and validate (first 5 records)
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
print("\nValidating embeddings by re-encoding content...")
for record in db_records[:5]:  # Limit validation for efficiency
    doc_id, filename, content, keywords, db_faiss_index = record
    if db_faiss_index is not None and db_faiss_index < num_vectors:
        # Re-encode document content
        reencoded_vector = model.encode([content], convert_to_tensor=True).cpu().numpy()[0]
        
        # Fetch the corresponding FAISS vector
        db_faiss_index = int(round(db_faiss_index))  # Round and cast again
        faiss_vector = vectors[db_faiss_index]
        
        # Compute difference
        diff = np.linalg.norm(reencoded_vector - faiss_vector)
        print(f"Document '{filename}' (ID: {doc_id}) - Embedding Difference: {diff:.6f}")

# Summary
if mismatches > 0:
    print(f"\nTotal mismatches found: {mismatches}")
else:
    print("\nNo mismatches found. FAISS and SQLite are synchronized.")

conn.close()

