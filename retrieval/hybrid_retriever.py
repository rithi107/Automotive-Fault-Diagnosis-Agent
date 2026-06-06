import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from retrieval.embedder import Embedder
from retrieval.bm25_retriever import BM25Retriever

load_dotenv()

COLLECTION_NAME = "obd_knowledge_base"

class HybridRetriever:
    def __init__(self):
        print("Initializing Hybrid Retriever...")
        self.embedder = Embedder()
        self.bm25 = BM25Retriever()
        self.qdrant = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=30
        )
        print("Hybrid Retriever ready.")
    
    def dense_search(self, query: str, top_k: int = 5) -> list:
        """Search using dense embeddings in Qdrant"""
        query_vector = self.embedder.embed(query)
        
        results = self.qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k
        ).points
        
        return [{
            "dtc_code": r.payload["dtc_code"],
            "fault_name": r.payload["fault_name"],
            "symptoms": r.payload["symptoms"],
            "causes": r.payload["causes"],
            "repair_steps": r.payload["repair_steps"],
            "text": r.payload["text"],
            "score": r.score,
            "source": "dense"
        } for r in results]
    
    def hybrid_search(self, query: str, top_k: int = 5) -> list:
        """Combine dense and BM25 results using Reciprocal Rank Fusion"""
        dense_results = self.dense_search(query, top_k=top_k)
        bm25_results = self.bm25.search(query, top_k=top_k)
        
        # Reciprocal Rank Fusion
        scores = {}
        k = 60  # RRF constant
        
        for rank, result in enumerate(dense_results):
            code = result["dtc_code"]
            scores[code] = scores.get(code, 0) + 1 / (k + rank + 1)
        
        for rank, result in enumerate(bm25_results):
            code = result["dtc_code"]
            scores[code] = scores.get(code, 0) + 1 / (k + rank + 1)
        
        # Merge all results
        all_results = {r["dtc_code"]: r for r in dense_results + bm25_results}
        
        # Sort by RRF score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        final_results = []
        for code, rrf_score in ranked:
            result = all_results[code]
            result["rrf_score"] = rrf_score
            final_results.append(result)
        
        return final_results