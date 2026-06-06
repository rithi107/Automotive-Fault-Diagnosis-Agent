import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.hybrid_retriever import HybridRetriever

retriever = HybridRetriever()

print("=== HYBRID SEARCH ===")
results = retriever.hybrid_search("P0301 cylinder misfire", top_k=3)
for r in results:
    print(f"Code: {r['dtc_code']}")
    print(f"Fault: {r['fault_name']}")
    print(f"RRF Score: {r['rrf_score']}")
    print("---")