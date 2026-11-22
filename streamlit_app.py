import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import pandas as pd
from openai import OpenAI
import anthropic
import chromadb
import os

st.title("ELG Chatbot Mockup")

# Create ChromaDB client
db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ChromaDB_for_ELG')
chroma_client = chromadb.PersistentClient(path=db_path)
collection = chroma_client.get_or_create_collection('ELGCollection')

# Initialize both clients so OpenAI can always handle embeddings
openai_client = OpenAI(api_key=st.secrets.OPENAI_API_KEY)
anthropic_client = anthropic.Anthropic(api_key=st.secrets.ANTHROPIC_API_KEY)

# Choose between ChatGPT and Claude
llm_option = st.sidebar.selectbox(
    'Choose LLM',
    ("ChatGPT", "Claude")
)

if st.sidebar.checkbox("Use advanced model"):
    if llm_option == "ChatGPT":
        selected_model = "gpt-5.4-mini"
    else:
        selected_model = "claude-sonnet-4-6"  # Updated to valid Anthropic model ID
else:
    if llm_option == "ChatGPT":
        selected_model = "gpt-5.4-nano"
    else:
        selected_model = "claude-haiku-4-5-20251001"


def add_to_collection(collection, text, doc_id, metadata):
    # Always use OpenAI for embeddings to keep vector space consistent
    response = openai_client.embeddings.create(
        input=text,
        model='text-embedding-3-small'
    )
    embedding = response.data[0].embedding

    collection.add(
        documents=[text],
        metadatas=[metadata],
        ids=[doc_id],
        embeddings=[embedding]
    )


def load_csv_to_collection(csv_path, collection):
    df = pd.read_csv(csv_path)
    
    texts = []
    doc_ids = []
    metadatas = []
    
    for index, row in df.iterrows():
        text_chunk = " | ".join([f"{col}: {str(val)}" for col, val in row.items()])
        texts.append(text_chunk)
        doc_ids.append(f"news_doc_{index}")
        metadatas.append({"source": "news.csv", "row": index})
    
    BATCH = 1000
    for i in range(0, len(texts), BATCH):
        batch_texts = texts[i:i+BATCH]
        batch_ids = doc_ids[i:i+BATCH]
        batch_meta = metadatas[i:i+BATCH]
        
        response = openai_client.embeddings.create(
            input=batch_texts,
            model='text-embedding-3-small'
        )
        embeddings = [item.embedding for item in response.data]
        
        collection.add(
            documents=batch_texts,
            metadatas=batch_meta,
            ids=batch_ids,
            embeddings=embeddings
        )


if 'ELG_VectorDB' not in st.session_state:
    if collection.count() == 0:
        csv_path = 'ELG_Database.csv'
        if os.path.exists(csv_path):
            with st.spinner("Loading and embedding rate cases and proceedings. This may take a moment..."):
                load_csv_to_collection(csv_path, collection)
        else:
            st.error(f"Could not find '{csv_path}'. Please ensure it is in the same directory.")

    st.session_state.ELG_VectorDB = collection

collection = st.session_state.ELG_VectorDB

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

system_prompt = """You are an expert regulatory analysis bot used by legal and compliance professionals to analyze rate case filings and proceedings.
You must base your answers and analysis strictly on the examples of rate cases provided in the Context below. Do not use outside knowledge, and do not hallucinate details not present in the text.

Rules:
- If the user asks to "Compare these rate cases" or "Summarize the rate cases", provide a structured comparison of the key elements (e.g., requested revenue changes, Return on Equity (ROE), accepted arguments, or final rulings) based solely on the provided examples.
- If the user asks a specific question or says "Find examples of [Topic/Argument]", extract and return the relevant details, precedents, or data specifically from the provided rate case context. 
- Always cite the specific rate case name or identifier when referencing data or arguments.
- Be professional, accurate, analytical, and concise."""

if prompt := st.chat_input("Ask me anything about a rate case or proceeding..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        # 1. Generate query embedding via OpenAI
        query_embedding_response = openai_client.embeddings.create(
            input=prompt,
            model='text-embedding-3-small'
        )
        query_embedding = query_embedding_response.data[0].embedding

        # 2. Retrieve relevant context from ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=10
        )
        
        context_docs = results['documents'][0] if results['documents'] else []
        context_text = "\n\n---\n\n".join(context_docs)
        user_content = f"Context:\n{context_text}\n\nUser Question:\n{prompt}"

        # 3. Call the selected LLM provider with its respective SDK syntax
        if llm_option == "ChatGPT":
            response = openai_client.chat.completions.create(
                model=selected_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ]
            )
            answer = response.choices[0].message.content
        else:
            response = anthropic_client.messages.create(
                model=selected_model,
                max_tokens=4096,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_content}
                ]
            )
            answer = response.content[0].text

        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})