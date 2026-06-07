import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import json

st.set_page_config(
    page_title="Automotive Fault Diagnosis Agent",
    page_icon="🔧",
    layout="wide"
)

st.title("🔧 Automotive Fault Diagnosis Agent")
st.markdown("Multi-agent LangGraph system for intelligent OBD-II fault diagnosis")

@st.cache_resource(show_spinner="Loading pipeline... please wait")
def load_pipeline():
    from pipeline.graph import build_graph
    return build_graph()

pipeline = load_pipeline()

# Input section
st.header("🚗 Input Fault Data")

col1, col2 = st.columns(2)

with col1:
    st.subheader("DTC Codes")
    dtc_input = st.text_input(
        "Enter DTC codes (comma separated)",
        placeholder="e.g. P0301, P0171"
    )

with col2:
    st.subheader("Freeze Frame Data (Optional)")
    rpm = st.text_input("RPM at fault", placeholder="e.g. 2500")
    coolant_temp = st.text_input("Coolant Temperature", placeholder="e.g. 90C")
    fuel_trim = st.text_input("Fuel Trim", placeholder="e.g. +15%")
    map_sensor = st.text_input("MAP Sensor", placeholder="e.g. 45kPa")

if st.button("🔍 Diagnose", type="primary"):
    if not dtc_input.strip():
        st.error("Please enter at least one DTC code.")
    else:
        dtc_codes = [code.strip().upper() for code in dtc_input.split(",")]
        
        freeze_frame = {}
        if rpm: freeze_frame["rpm"] = rpm
        if coolant_temp: freeze_frame["coolant_temp"] = coolant_temp
        if fuel_trim: freeze_frame["fuel_trim"] = fuel_trim
        if map_sensor: freeze_frame["map_sensor"] = map_sensor
        
        with st.spinner("🤖 Agents are diagnosing the fault..."):
            result = pipeline.invoke({
                "dtc_codes": dtc_codes,
                "freeze_frame": freeze_frame,
                "retrieved_docs": [],
                "diagnosis": "",
                "response": {}
            })
        
        response = result["response"]
        
        st.header("📋 Diagnosis Report")
        
        severity = response.get("severity", "Unknown")
        severity_colors = {
            "Low": "🟢",
            "Medium": "🟡",
            "High": "🟠",
            "Critical": "🔴"
        }
        badge = severity_colors.get(severity, "⚪")
        
        st.markdown(f"### {badge} Severity: {severity}")
        st.markdown(f"**{response.get('severity_explanation', '')}**")
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Fault Summary")
            st.info(response.get("fault_summary", "N/A"))
            
            st.subheader("⚠️ Immediate Action")
            st.warning(response.get("immediate_action", "N/A"))
            
            st.subheader("🔩 Estimated Repair")
            st.write(response.get("estimated_repair", "N/A"))
        
        with col2:
            st.subheader("🔍 Root Causes")
            for i, cause in enumerate(response.get("root_causes", []), 1):
                st.write(f"{i}. {cause}")
            
            st.subheader("🛠️ Diagnostic Steps")
            for i, step in enumerate(response.get("diagnostic_steps", []), 1):
                st.write(f"{i}. {step}")
        
        st.divider()
        
        with st.expander("📄 View Raw JSON Response"):
            st.json(response)
        
        with st.expander("📚 View Retrieved Knowledge Base Documents"):
            for doc in result["retrieved_docs"]:
                st.markdown(f"**{doc['dtc_code']}** — {doc['fault_name']}")
                st.divider()