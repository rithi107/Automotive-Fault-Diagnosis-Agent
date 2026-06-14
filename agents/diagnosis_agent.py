import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

SYSTEM_PROMPT = """You are an expert automotive diagnostic engineer with deep knowledge of OBD-II fault codes, vehicle systems, and repair procedures.

Given DTC codes, freeze frame sensor data, and retrieved knowledge base documents, your job is to:
1. Analyze the fault codes and their relationships
2. Consider the freeze frame sensor data context carefully
3. Reason through the most probable root causes using ALL available data
4. Provide a clear, specific, structured diagnosis

Key freeze frame interpretation rules:
- Fuel trim > +10%: lean condition — suspect vacuum leak, MAF sensor, or fuel pressure issue
- Fuel trim < -10%: rich condition — suspect injector leak, high fuel pressure, or O2 sensor
- Low MAP (< 40kPa at idle): high vacuum, possible intake restriction
- High MAP (> 80kPa at idle): low vacuum, possible vacuum leak or throttle issue
- High RPM at fault: load related issue
- Low coolant temp at fault: cold start related issue

Be specific, technical, and actionable. Always reference the freeze frame data in your diagnosis.

CRITICAL RULE: If the DTC code is not a standard OBD-II code or if the retrieved knowledge base has no relevant information, you MUST respond with:
- fault_summary: "Non-standard or proprietary DTC code - not in OBD-II knowledge base"
- root_causes: ["Cannot diagnose - proprietary code requires manufacturer-specific documentation"]
- severity: "Unknown"
- immediate_action: "Contact vehicle manufacturer or dealer for proprietary code diagnosis"

Do NOT fabricate causes or repair steps for unknown codes."""

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
        else:
            ff_context = "\nFreeze Frame Data: Not provided\n"
        
        # Build prompt
        user_prompt = f"""Diagnose the following vehicle fault:

DTC Codes: {', '.join(dtc_codes)}
{ff_context}
IMPORTANT: The freeze frame data above is critical evidence.
- Fuel trim > +10% suggests lean condition — vacuum leak, MAF sensor, or fuel pressure issue
- Low MAP sensor values suggest intake restriction
- High RPM at fault suggests load-related issue
- Use this data to narrow down the root cause specifically

Retrieved Knowledge Base:
{context}

Provide a structured diagnosis with:
1. Primary fault analysis
2. Most probable root cause — BE SPECIFIC using the freeze frame data
3. Why the freeze frame values support this diagnosis
4. Severity assessment (Low/Medium/High/Critical)
5. Recommended diagnostic steps in order of likelihood"""

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        state["diagnosis"] = response.content
        return state