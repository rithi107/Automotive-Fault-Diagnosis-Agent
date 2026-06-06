from retrieval.hybrid_retriever import HybridRetriever
from typing import TypedDict, List

class AgentState(TypedDict):
    dtc_codes: List[str]
    freeze_frame: dict
    retrieved_docs: List[dict]
    diagnosis: str
    response: dict

class RetrievalAgent:
    def __init__(self):
        self.retriever = HybridRetriever()
    
    def run(self, state: AgentState) -> AgentState:
        """Retrieve relevant docs for given DTC codes"""
        dtc_codes = state["dtc_codes"]
        freeze_frame = state.get("freeze_frame", {})
        
        # Build query from DTC codes + freeze frame context
        query_parts = dtc_codes.copy()
        
        if freeze_frame:
            if freeze_frame.get("rpm"):
                query_parts.append(f"RPM {freeze_frame['rpm']}")
            if freeze_frame.get("coolant_temp"):
                query_parts.append(f"coolant temperature {freeze_frame['coolant_temp']}")
            if freeze_frame.get("fuel_trim"):
                query_parts.append(f"fuel trim {freeze_frame['fuel_trim']}")
        
        query = " ".join(query_parts)
        print(f"Retrieval query: {query}")
        
        # Retrieve for each DTC code separately then combine
        all_docs = []
        seen_codes = set()
        
        for code in dtc_codes:
            results = self.retriever.hybrid_search(code, top_k=3)
            for doc in results:
                if doc["dtc_code"] not in seen_codes:
                    all_docs.append(doc)
                    seen_codes.add(doc["dtc_code"])
        
        # Also search with full query for context
        context_results = self.retriever.hybrid_search(query, top_k=3)
        for doc in context_results:
            if doc["dtc_code"] not in seen_codes:
                all_docs.append(doc)
                seen_codes.add(doc["dtc_code"])
        
        print(f"Retrieved {len(all_docs)} relevant documents")
        state["retrieved_docs"] = all_docs
        return state