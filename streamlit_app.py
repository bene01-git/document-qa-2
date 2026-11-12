import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import pandas as pd
import json
from openai import OpenAI
import chromadb
from pathlib import Path
from bs4 import BeautifulSoup
import os

st.title("ELG Chatbot Mockup")

# Create ChromaDB client
db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ChromaDB_for_ELG')
chroma_client = chromadb.PersistentClient(path=db_path)
collection = chroma_client.get_or_create_collection('ELGCollection')

# Create OpenAI client
if 'open_ai_client' not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=st.secrets.OPENAI_API_KEY)

def add_to_collection(collection, text, doc_id, metadata):
    # Create embedding
    client = st.session_state.openai_client
    response = client.embeddings.create(
        input=text,
        model='text-embedding-3-small'
    )

    # Get embedding
    embedding = response.data[0].embedding

    # Add embedding, document, and metadata to ChromaDB
    collection.add(
        documents=[text],
        metadatas=[metadata],
        ids=[doc_id],
        embeddings=[embedding]
    )

def load_csv_to_collection(csv_path, collection):
    df = pd.read_csv(csv_path)
    # Combine all columns in a row into a single searchable text string
    for index, row in df.iterrows():
        text_chunk = " | ".join([f"{col}: {str(val)}" for col, val in row.items()])
        doc_id = f"news_doc_{index}"
        metadata = {"source": "news.csv", "row": index}
        add_to_collection(collection, text_chunk, doc_id, metadata)

if 'HW7_VectorDB' not in st.session_state:
    # Check if collection is empty and load the news.csv file
    if collection.count() == 0:
        csv_path = 'ELG_Database.csv'
        if os.path.exists(csv_path):
            with st.spinner("Loading and embedding news articles. This may take a moment..."):
                load_csv_to_collection(csv_path, collection)
        else:
            st.error(f"Could not find '{csv_path}'. Please ensure it is in the same directory.")

    st.session_state.ELG_VectorDB = collection

collection = st.session_state.ELG_VectorDB

model = st.sidebar.selectbox('Which model?', ('mini', 'nano'))

if model == "mini":
    model_to_use = "gpt-5-mini"
if model == "nano":
    model_to_use = "gpt-5-nano"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me about the news, or ask to find the most interesting news..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        client = st.session_state.openai_client
        
        query_embedding_response = client.embeddings.create(
            input=prompt,
            model='text-embedding-3-small'
        )
        query_embedding = query_embedding_response.data[0].embedding

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=10 # Top 10 relevant items to query
        )
        
        context_docs = results['documents'][0] if results['documents'] else []
        context_text = "\n\n---\n\n".join(context_docs)

        system_prompt = """You are an expert regulatory analysis bot used by legal and compliance professionals to analyze rate case filings and proceedings.
        You must base your answers and analysis strictly on the examples of rate cases provided in the Context below. Do not use outside knowledge, and do not hallucinate details not present in the text.

        Rules:
        - If the user asks to "Compare these rate cases" or "Summarize the rate cases", provide a structured comparison of the key elements (e.g., requested revenue changes, Return on Equity (ROE), accepted arguments, or final rulings) based solely on the provided examples.
        - If the user asks a specific question or says "Find examples of [Topic/Argument]", extract and return the relevant details, precedents, or data specifically from the provided rate case context. 
        - Always cite the specific rate case name or identifier when referencing data or arguments.
        - Be professional, accurate, analytical, and concise."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context_text}\n\nUser Question:\n{prompt}"}
        ]

        response = client.chat.completions.create(
            model=model_to_use,
            messages=messages
        )
        
        answer = response.choices[0].message.content
        st.markdown(answer)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})