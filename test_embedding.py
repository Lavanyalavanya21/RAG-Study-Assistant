from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "Information retrieval is about searching data",
    "Search engines use indexing",
    "I like pizza"
]

embeddings = model.encode(sentences)

print("Shape:", embeddings.shape)

# Check similarity
similarity = np.dot(embeddings[0], embeddings[1])
print("Similarity (related):", similarity)

similarity2 = np.dot(embeddings[0], embeddings[2])
print("Similarity (unrelated):", similarity2)