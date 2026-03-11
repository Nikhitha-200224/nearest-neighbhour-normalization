# Nearest Neighbor Normalization using Faiss

import numpy as np
import faiss

# Sample dataset (image/text embeddings)
data = np.random.random((100, 128)).astype('float32')

# Create FAISS index
index = faiss.IndexFlatL2(128)
index.add(data)

# Query vector
query = np.random.random((1, 128)).astype('float32')

# Search nearest neighbors
k = 5
distances, indices = index.search(query, k)

print("Nearest Neighbor Indices:", indices)
print("Distances:", distances)

# Normalization
normalized_scores = distances / np.max(distances)

print("Normalized Scores:", normalized_scores)
