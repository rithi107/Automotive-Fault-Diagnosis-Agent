import json
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "obd_knowledge_base"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_SIZE = 384  # all-MiniLM-L6-v2 output dimension
BATCH_SIZE = 20

def load_merged_data(path: str = "data/merged_obd.json") -> list:
    with open(path, "r") as f:
        data = json.load(f)
    # Filter only records with meaningful content
    filtered = [r for r in data if r["causes"] or r["symptoms"]]
    print(f"Loaded {len(filtered)} records with rich content")
    return filtered

def init_qdrant() -> QdrantClient:
    
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=60  # Add this
    )
    
    collections = [c.name for c in client.get_collections().collections]
    
    if COLLECTION_NAME not in collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )
        print(f"Collection '{COLLECTION_NAME}' created.")
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists.")
    
    return client

def embed_and_ingest(docs: list, client: QdrantClient):
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    print(f"Embedding {len(docs)} documents...")
    texts = [doc['text'] for doc in docs]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    
    print("Ingesting into Qdrant in batches...")
    for i in tqdm(range(0, len(docs), BATCH_SIZE), desc="Uploading batches"):
        batch_docs = docs[i:i + BATCH_SIZE]
        batch_embeddings = embeddings[i:i + BATCH_SIZE]
        
        points = []
        for j, (doc, embedding) in enumerate(zip(batch_docs, batch_embeddings)):
            points.append(PointStruct(
                id=i + j,
                vector=embedding.tolist(),
                payload={
                    "dtc_code": doc["dtc_code"],
                    "fault_name": doc["fault_name"],
                    "symptoms": doc["symptoms"],
                    "causes": doc["causes"],
                    "repair_steps": doc["repair_steps"],
                    "text": doc["text"]
                }
            ))
        
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
    
    print(f"Successfully ingested {len(docs)} documents.")

if __name__ == "__main__":
    docs = load_merged_data()
    client = init_qdrant()
    embed_and_ingest(docs, client)
    print("Ingestion complete!")