import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from pinecone import ServerlessSpec

import logging
import sys
import os

# from google.colab import userdata

import qdrant_client
from qdrant_client import models
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.groq import Groq # deep seek r1 implementation

model = SentenceTransformer('all-MiniLM-L6-v2')

from dotenv import load_dotenv

load_dotenv()  # Load .env variables

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "bhagwat-gita"

index = pc.Index(index_name)


def query_verse(query, k=5):
    # Generate embedding for the query
    query_embedding = model.encode([query])[0]  # Single query embedding

    # Search the Pinecone index
    results = index.query(
        vector=query_embedding.tolist(),
        top_k=k,
        include_metadata=True
    )

    # Extract the corresponding verses
    similar_verses = []
    for match in results['matches']:
        similar_verses.append({
            "verse_text": match['metadata']['combined_text'],
            "similarity_score": match['score']
        })

    return similar_verses



# Example query
query = "i am very much deppressed. which is the best shlokas for me to ovecome this"
similar_verses = query_verse(query, k=5)

# Print the most similar verses and their similarity scores
# for i, verse in enumerate(similar_verses):
#     print(f"Verse {i + 1}: {verse['verse_text']}")
#     print(f"Similarity Score: {verse['similarity_score']}\n")


llm = Groq(model="deepseek-r1-distill-llama-70b")


from llama_index.core import ChatPromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole

message_templates = [
    ChatMessage(
        content="""
        You are an expert ancient assistant who is well verse in Bhagavad-gita.
        You are Multilingual, you understand English, Hindi and Sanskrit.

        ⚠️ IMPORTANT: You must NEVER answer questions that are NOT related to the Bhagavad Gita.
        - If a question is not about the Bhagavad Gita, simply reply with:
          "I don't know. Not enough information received."
        - If the provided verses do not contain relevant information, reply with:
          "I don't know. Not enough information received."
        - Do NOT make up answers. Stay strictly within the given context.
        """,
        role=MessageRole.SYSTEM),
    ChatMessage(
        content="""
        We have provided context information below.
        {similar_verses}
        ---------------------
        Given this information, please answer the question: {query}
        ---------------------
        If the question is not from the provided context or just unrelated from bhagwat gita i say again if a question just 1% unrelated, say `I don't know. Not enough information recieved.`
        """,
        role=MessageRole.USER,
    ),
]


def pipeline(query):
    # R - Retriver
    relevant_documents = query_verse(query)
    context = [doc['verse_text'] for doc in relevant_documents]
    context = "\n".join(context)

    # A - Augment
    chat_template = ChatPromptTemplate(message_templates=message_templates)

    # G - Generate
    response = llm.complete(
        chat_template.format(
            context=context,
            query=query)
    )
    return response
