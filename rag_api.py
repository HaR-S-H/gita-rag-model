from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from rag_model import pipeline  # Import your RAG function

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
def get_rag_response(request: QueryRequest):
    response = pipeline(request)
    return {"response": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
