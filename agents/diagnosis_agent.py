import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

SYSTEM_PROMPT = """You are an expert automotive diagnostic engineer with deep knowledge of OBD-II fault codes, vehicle systems, and repair procedures.

Given DTC codes, freeze frame sensor data, and retrieved knowledge base documents, your job is to:
1. Analyze the fault codes and their relationships
2. Consider the freeze frame sensor data context
3. Reason through the most probable root causes
4. Provide a clear, structured diagnosis

Be specific, technical, and actionable. Focus on the most likely root causes given all available information."""

class DiagnosisAgent:
    def __init__(self):
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.1
        )
    
    def run(self, state: dict) -> dict:
        """Reason through diagnosis based on retrieved docs"""
        dtc_codes = state["dtc_codes"]
        freeze_frame = state.get("freeze_frame", {})
        retrieved_docs = state["retrieved_docs"]
        
        # Build context from retrieved docs
        context = ""
        for doc in retrieved_docs:
            context += f"\n--- {doc['dtc_code']}: {doc['fault_name']} ---\n"
            if doc["symptoms"]:
                context += f"Symptoms: {', '.join(doc['symptoms'])}\n"
            if doc["causes"]:
                context += f"Causes: {', '.join(doc['causes'])}\n"
            if doc["repair_steps"]:
                context += f"Repair Steps: {', '.join(doc['repair_steps'])}\n"
        
        # Build freeze frame context
        ff_context = ""
        if freeze_frame:
            ff_context = f"\nFreeze Frame Data at time of fault:\n"
            for key, value in freeze_frame.items():
                ff_context += f"  {key}: {value}\n"
        
        # Build prompt
        user_prompt = f"""Diagnose the following vehicle fault:

DTC Codes: {', '.join(dtc_codes)}
{ff_context}
Retrieved Knowledge Base:
{context}

Provide a structured diagnosis with:
1. Primary fault analysis
2. Most probable root cause(s)
3. How the freeze frame data supports the diagnosis
4. Severity assessment (Low/Medium/High/Critical)
5. Recommended diagnostic steps in order"""

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        state["diagnosis"] = response.content
        return state