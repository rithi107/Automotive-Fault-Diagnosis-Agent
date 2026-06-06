from rank_bm25 import BM25Okapi
import json

class BM25Retriever:
    def __init__(self, data_path: str = "data/merged_obd.json"):
        print("Loading BM25 index...")
        with open(data_path, "r") as f:
            self.docs = json.load(f)
        
        # Filter only rich records
        self.docs = [d for d in self.docs if d["causes"] or d["symptoms"]]
        
        # Tokenize corpus
        corpus = [doc["text"].lower().split() for doc in self.docs]
        self.bm25 = BM25Okapi(corpus)
        print(f"BM25 index built with {len(self.docs)} documents.")
    
    def search(self, query: str, top_k: int = 5) -> list:
        """Search using BM25 keyword matching"""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]
        
        results = []
        for idx in top_indices:
            doc = self.docs[idx]
            results.append({
                "dtc_code": doc["dtc_code"],
                "fault_name": doc["fault_name"],
                "symptoms": doc["symptoms"],
                "causes": doc["causes"],
                "repair_steps": doc["repair_steps"],
                "text": doc["text"],
                "score": float(scores[idx]),
                "source": "bm25"
            })
        
        return results