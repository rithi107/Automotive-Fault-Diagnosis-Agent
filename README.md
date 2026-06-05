# 🔧 Automotive Fault Diagnosis Agent

A multi-agent LangGraph system that diagnoses automotive faults from DTC codes 
and freeze frame sensor data using hybrid RAG retrieval over OBD-II knowledge bases.

---

## 🚗 Problem Statement

When a vehicle throws a fault code, mechanics and engineers spend significant time 
manually cross-referencing DTC codes across repair manuals, technical service 
bulletins, and past repair histories.

Simply asking an LLM "what is P0301?" gives a generic answer. But real diagnosis requires reasoning over:
- **Freeze frame sensor data** (RPM, coolant temp, fuel trim, MAP at fault time)
- **Multi-fault correlation** (P0301 + P0171 + P0174 together = different diagnosis)
- **Manufacturer and model-specific repair knowledge**
- **Structured actionable output** a mechanic can actually use

This system automates real diagnosis — not just code lookup — using a multi-agent 
AI pipeline that retrieves specific fault knowledge and reasons through probable 
root causes, severity, and recommended repair actions.

---

## 🏗️ Architecture

| Stage | Component | Description |
|-------|-----------|-------------|
| Input | User | DTC codes + freeze frame sensor data |
| Step 1 | Retrieval Agent | Hybrid search (Dense + BM25) over Qdrant |
| Step 2 | Diagnosis Agent | Multi-fault reasoning via LangGraph |
| Step 3 | Response Agent | Root cause + severity + repair steps |
| Output | Streamlit UI | Structured diagnosis displayed to user |

---

## ⚙️ Tech Stack

| Component | Technology |
|-----------|------------|
| Agent Orchestration | LangGraph |
| Vector Database | Qdrant |
| Hybrid Retrieval | Dense Embeddings + BM25 |
| Embeddings | sentence-transformers |
| LLM | Llama 3.3 via Groq |
| UI | Streamlit |
| Language | Python 3.11+ |

---

## 🔄 How It Works

1. User inputs one or more DTC codes (e.g. `P0301`, `P0171`) and optional freeze frame sensor data via Streamlit UI
2. **Retrieval Agent** performs hybrid search — dense semantic search + BM25 keyword search — over the OBD-II knowledge base stored in Qdrant
3. **Diagnosis Agent** receives retrieved context and reasons through probable root causes using LangGraph, accounting for multi-fault correlations and sensor data
4. **Response Agent** formats structured output — root cause, severity rating, recommended repair steps, parts likely needed
5. Results displayed in a clean Streamlit interface

---

## 💡 Why Not Just Use an LLM Directly?

| Capability | LLM Alone | This System |
|------------|-----------|-------------|
| Generic DTC explanation | ✅ | ✅ |
| Freeze frame data reasoning | ❌ | ✅ |
| Multi-fault correlation | ❌ | ✅ |
| Manufacturer-specific knowledge | ❌ | ✅ |
| Structured repair output | ❌ | ✅ |

---

## 📁 Project Structure

| Path | Description |
|------|-------------|
| `agents/retrieval_agent.py` | Hybrid search over Qdrant |
| `agents/diagnosis_agent.py` | Root cause reasoning |
| `agents/response_agent.py` | Structured output formatting |
| `pipeline/graph.py` | LangGraph orchestration |
| `retrieval/embedder.py` | sentence-transformers embeddings |
| `retrieval/bm25_retriever.py` | BM25 keyword retrieval |
| `retrieval/hybrid_retriever.py` | Combined hybrid retrieval |
| `data/ingest.py` | OBD-II knowledge base ingestion |
| `ui/app.py` | Streamlit UI |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Qdrant running locally or via Qdrant Cloud free tier
- Groq API key (free at console.groq.com)

### Installation

```bash
git clone https://github.com/rithi107/Automotive-Fault-Diagnosis-Agent.git
cd Automotive-Fault-Diagnosis-Agent
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # Add your API keys
```

### Run Qdrant Locally

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### Run the App

```bash
streamlit run ui/app.py
```

---

## 📸 Demo

> Coming soon — UI screenshot and walkthrough GIF

---

## 🗺️ Roadmap

- [x] Project setup and architecture
- [ ] OBD-II knowledge base ingestion pipeline
- [ ] Qdrant hybrid retrieval setup
- [ ] LangGraph multi-agent pipeline
- [ ] Streamlit UI
- [ ] Demo GIF and screenshots
- [ ] Docker support

---

## 🙋‍♀️ Author

Rithika
Automotive BSW + GenAI Engineer
