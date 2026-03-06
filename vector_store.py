from sentence_transformers import SentenceTransformer
import faiss
import pickle

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAQ data
with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Split questions
docs = text.split("\n\n")

# Convert to embeddings
embeddings = model.encode(docs)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save index
faiss.write_index(index, "faiss_index/index.faiss")

# Save texts
with open("faiss_index/texts.pkl", "wb") as f:
    pickle.dump(docs, f)

print("Vector database created successfully!")