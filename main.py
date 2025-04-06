import os
import faiss
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from typing import Optional
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

# Load env variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY in environment variables.")

# Configure Gemini and models
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# Load models and data
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
df = pd.read_csv("shl_catalog_with_summaries.csv")
index = faiss.read_index("shl_assessments_index.faiss")

app = FastAPI(title="SHL Assessment Search API")

# LLM preprocessing
def llm_shorten_query(query: str) -> str:
    prompt = "I want to search my vector database with the query you will recieve, your task is to just summarize the query (maximum 10 words) only retaining technical skills. Query: "
    try:
        response = model.generate_content(prompt + query)
        return response.text.strip()
    except Exception:
        return query

# Retrieval logic
def retrieve_assessments(query: str, k: int = 10, max_duration: Optional[int] = None):
    query_lower = query.lower()
    wants_flexible = any(x in query_lower for x in ["untimed", "variable", "flexible"])
    processed_query = llm_shorten_query(query)
    query_embedding = embedding_model.encode([processed_query], show_progress_bar=False)[0]
    query_embedding = np.array([query_embedding], dtype="float32")
    distances, indices = index.search(query_embedding, k * 2)
    results = df.iloc[indices[0]].copy()
    results["similarity_score"] = 1 - distances[0] / 2
    if max_duration is not None or wants_flexible:
        filtered = []
        for _, row in results.iterrows():
            duration = row["Assessment Length Parsed"]
            if pd.isna(duration):
                filtered.append(row)
            elif duration == "flexible duration" and wants_flexible:
                filtered.append(row)
            elif isinstance(duration, float) and max_duration is not None and duration <= max_duration:
                filtered.append(row)
        results = pd.DataFrame(filtered) if filtered else results
    results = results.rename(columns={
        "Pre-packaged Job Solutions": "Assessment Name",
        "Assessment Length": "Duration"
    })
    return results[[
        "Assessment Name",
        "URL",
        "Remote Testing (y/n)",
        "Adaptive/IRT (y/n)",
        "Duration",
        "Test Type"
    ]].head(k).to_dict(orient="records")

# FastAPI endpoint
@app.get("/recommend")  # Changed to /recommend for consistency
def recommend(query: str = Query(..., description="Your search text"),
              k: int = Query(5, description="Number of results to return"),
              max_duration: Optional[int] = Query(None, description="Max duration filter in minutes")):
    try:
        results = retrieve_assessments(query, k=k, max_duration=max_duration)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Render runs via uvicorn main:app --host 0.0.0.0 --port $PORT
