import faiss
import sqlite3
import numpy as np

# Paths to the FAISS index file and SQLite database
faiss_index_path = 'data/faiss_index.bin'
db_path = 'data/pdf_index.db'

# Load the FAISS index
faiss_index = faiss.read_index(faiss_index_path)

# Get basic properties of the FAISS index
num_vectors = faiss_index.ntotal  # Total number of vectors in the index
dimension = faiss_index.d         # Dimensionality of each vector

print(f"FAISS Index Loaded:")
print(f"- Number of Vectors: {num_vectors}")
print(f"- Dimension of Each Vector: {dimension}")

# Retrieve all vectors from the FAISS index
vectors = np.zeros((num_vectors, dimension), dtype=np.float32)
faiss_index.reconstruct_n(0, num_vectors, vectors)

# Connect to the SQLite database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Retrieve all records from the SQLite database
cursor.execute("SELECT id, filename, content, keywords, faiss_index FROM documents")
db_records = cursor.fetchall()

# Perform a complete comparison
print("\nComparing FAISS Index with SQLite Records:")
mismatches = 0
if len(db_records) != num_vectors:
    print(f"Warning: Mismatch in the number of records! SQLite: {len(db_records)}, FAISS: {num_vectors}")

for record in db_records:
    doc_id, filename, content, keywords, db_faiss_index = record

    if db_faiss_index < 0 or db_faiss_index >= num_vectors:
        print(f"Error: FAISS index for document '{filename}' (ID: {doc_id}) is out of bounds: {db_faiss_index}")
        mismatches += 1
        continue

    # Compare embeddings
    faiss_vector = vectors[db_faiss_index]
    print(f"Document '{filename}' (SQLite ID: {doc_id}, FAISS Index: {db_faiss_index})")
    print(f"- Keywords: {keywords[:100]}")
    print(f"- FAISS Vector Sample: {faiss_vector[:5]}")  # Display first 5 dimensions for readability

# Optional: Compare by manually encoding the content (requires the same model)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

print("\nValidating embeddings by re-encoding content...")
for record in db_records[:5]:  # Limit validation to a few records for efficiency
    doc_id, filename, content, keywords, db_faiss_index = record
    if db_faiss_index >= 0 and db_faiss_index < num_vectors:
        # Re-encode the content
        reencoded_vector = model.encode([content], convert_to_tensor=True).cpu().numpy()[0]

        # Compute the difference between re-encoded and FAISS-stored vectors
        faiss_vector = vectors[db_faiss_index]
        diff = np.linalg.norm(reencoded_vector - faiss_vector)
        print(f"Document '{filename}' (ID: {doc_id}) - Embedding Difference: {diff:.6f}")
        if diff > 1e-3:
            print(f"Warning: Significant difference detected for document '{filename}'")

# Close the database connection
conn.close()

# Save vectors and mismatched records for further analysis
if mismatches > 0:
    print(f"\nTotal mismatches found: {mismatches}")
    with open("mismatch_log.txt", "w") as f:
        f.write(f"Total mismatches: {mismatches}\n")
    print("Mismatches saved to mismatch_log.txt")
else:
    print("\nNo mismatches found. FAISS and SQLite are synchronized.")

