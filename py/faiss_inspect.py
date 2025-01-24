import faiss
import numpy as np
import hashlib

# Path to the FAISS index file
faiss_index_path = 'data/faiss_index.bin'

# Load the FAISS index
faiss_index = faiss.read_index(faiss_index_path)

# Get basic properties
num_vectors = faiss_index.ntotal  # Total number of vectors in the index
dimension = faiss_index.d         # Dimensionality of each vector

print(f"FAISS Index Loaded:")
print(f"- Number of Vectors: {num_vectors}")
print(f"- Dimension of Each Vector: {dimension}")

# Hashing function to convert FAISS index to a unique hash
def hash_index(idx):
    """Hash the FAISS index to compare with stored hashes."""
    return hashlib.sha256(str(idx).encode('utf-8')).hexdigest()

# Retrieve vectors for inspection
vectors = np.zeros((num_vectors, dimension), dtype=np.float32)
faiss_index.reconstruct_n(0, num_vectors, vectors)

# Display the vectors (for debugging)
print("\nSample Vectors:")
print(vectors[:5])  # Print the first 5 vectors (or adjust as needed)

# Verify vector properties
if num_vectors > 0:
    print("\nFirst Vector Details:")
    print(f"Vector[0]: {vectors[0]}")
else:
    print("The FAISS index is empty.")

# Optional: Save vectors to a file for further inspection
np.savetxt("data/faiss_vectors.txt", vectors, delimiter=",")
print("All vectors have been saved to data/faiss_vectors.txt for detailed inspection.")

# Optional: Display corresponding FAISS index hashes for inspection
print("\nFAISS Index Hashes:")
for i in range(min(10, num_vectors)):  # Display first 10 FAISS index hashes
    faiss_index_hash = hash_index(i)
    print(f"Index {i} - Hash: {faiss_index_hash}")

