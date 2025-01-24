import faiss
import numpy as np

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

# Retrieve vectors for inspection
# FAISS doesn't directly store IDs unless an ID map is used.
# Use the range of indices as they correspond to the vectors' positions in the index.
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

