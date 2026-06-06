import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
import json

load_dotenv()

SYSTEM_PROMPT = """You are an automotive diagnostic report generator. 
Convert diagnostic analysis into a clean structured JSON report.
Always respond with valid JSON only, no other text."""

class ResponseAgent:
    def __init__(self):
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0
        )
    
    def run(self, state: dict) -> dict:
        """Format diagnosis into structured output"""
        diagnosis = state["diagnosis"]
        dtc_codes = state["dtc_codes"]
        
        user_prompt = f"""Convert this automotive diagnosis into a structured JSON report:

DTC Codes: {', '.join(dtc_codes)}

Diagnosis:
{diagnosis}

Return a JSON object with exactly these fields:
{{
    "dtc_codes": ["list of codes"],
    "fault_summary": "one line summary",
    "root_causes": ["list of probable root causes"],
    "severity": "Low/Medium/High/Critical",
    "severity_explanation": "why this severity",
    "diagnostic_steps": ["ordered list of steps"],
    "estimated_repair": "brief estimate of repair complexity",
    "immediate_action": "what driver should do right now"
}}"""

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        # Parse JSON response
        try:
            clean = response.content.strip()
            if "```json" in clean:
                clean = clean.split("```json")[1].split("```")[0].strip()
            elif "```" in clean:
                clean = clean.split("```")[1].split("```")[0].strip()
            
            state["response"] = json.loads(clean)
        except Exception as e:
            print(f"JSON parsing error: {e}")
            state["response"] = {"raw": response.content}
        
        return state