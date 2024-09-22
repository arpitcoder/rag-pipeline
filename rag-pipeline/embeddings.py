# embeddings.py

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load pre-trained embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def create_document_embeddings(documents):
    embeddings = model.encode(documents)
    return embeddings

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # Use L2 distance for similarity
    index.add(embeddings)
    return index

