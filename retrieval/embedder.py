from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"

class Embedder:
    def __init__(self):
        print("Loading embedding model...")
        self.model = SentenceTransformer(MODEL_NAME)
    
    def embed(self, text: str) -> list:
        """Embed a single text"""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def embed_batch(self, texts: list) -> np.ndarray:
        """Embed a list of texts"""
        return self.model.encode(texts, convert_to_numpy=True)