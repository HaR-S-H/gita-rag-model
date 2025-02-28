from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()
from rag_model import pipeline  # Import your RAG function

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
def get_rag_response(request: QueryRequest):
    response = pipeline(request)
    return {"response": response}

# if __name__ == "__main__":
#     port=int(os.getenv("PORT",8000))
#     print(f"Running on PORT:{port}")
#     uvicorn.run(app, host="0.0.0.0", port=port)
