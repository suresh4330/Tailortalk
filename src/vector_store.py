import os
import faiss
import numpy as np
import json

class SareeVectorStore:
    def __init__(self, dimension=512):
        """
        Initializes FAISS with Inner Product (for Cosine Similarity on normalized vectors).
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata = []
        
    def add_embeddings(self, embeddings, metadata_list):
        """
        Adds normalized embeddings and their metadata to the index.
        """
        if len(embeddings) != len(metadata_list):
            raise ValueError("Number of embeddings must match number of metadata records")
            
        # Ensure float32 for FAISS
        embeddings_np = np.array(embeddings).astype('float32')
        self.index.add(embeddings_np)
        self.metadata.extend(metadata_list)
        
    def search(self, query_embedding, top_k=5):
        """
        Searches the index for the most similar vectors.
        """
        query_np = np.array([query_embedding]).astype('float32')
        
        # D is distances (similarity scores since it's Inner Product), I is indices
        D, I = self.index.search(query_np, top_k)
        
        results = []
        for j in range(len(I[0])):
            idx = I[0][j]
            if idx != -1: # Valid index
                score = float(D[0][j])
                meta = self.metadata[idx]
                results.append({
                    "image_id": meta.get("SKU", str(idx)),
                    "image_path": meta.get("local_image_path", ""),
                    "similarity_score": score,
                    "metadata": meta
                })
        return results

    def save(self, index_path, metadata_path):
        """Saves the index and metadata to disk."""
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f)
            
    def load(self, index_path, metadata_path):
        """Loads the index and metadata from disk."""
        self.index = faiss.read_index(index_path)
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        self.dimension = self.index.d
