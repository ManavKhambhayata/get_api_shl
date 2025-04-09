import os
import faiss
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Request model for POST
class QueryRequest(BaseModel):
    query: str

# LLM preprocessing
def llm_shorten_query(query: str) -> str:
    prompt = "Summarize query (max 10 words) retaining technical skills: "
    try:
        response = model.generate_content(prompt + query)
        return response.text.strip()
    except Exception:
        return query

# Retrieval logic (unchanged)
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
    "Test Type",
    "Description"  # Added this
]].head(k).to_dict(orient="records")

@app.post("/recommend")
def recommend(request: QueryRequest):
    try:
        results = retrieve_assessments(request.query, k=10)
        # Mapping of abbreviations to full test type descriptions
        test_type_map = {
            "A": "Ability & Aptitude",
            "B": "Biodata & Situational Judgement",
            "C": "Competencies",
            "D": "Development & 360",
            "E": "Assessment Exercises",
            "K": "Knowledge & Skills",
            "P": "Personality & Behavior",
            "S": "Simulations"
        }
        formatted_results = [
        {
            "url": result["URL"],
            "adaptive_support": result["Adaptive/IRT (y/n)"],
            "description": result["Description"],  # Now works with updated return
            "duration": result["Duration"],
            "remote_support": result["Remote Testing (y/n)"],
            "test_type": [test_type_map.get(abbrev.strip(), abbrev.strip()) 
                          for abbrev in result["Test Type"].split()] 
                          if pd.notna(result["Test Type"]) else []
        }
        for result in results
    ]
    return {"recommended_assessments": formatted_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# Render runs via uvicorn app:app --host 0.0.0.0 --port $PORT
