from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from huggingface_hub import InferenceClient
from langchain.prompts import PromptTemplate
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from dotenv import load_dotenv
import os
import re
import streamlit as st

# Load environment variables
load_dotenv()

# Load API keys
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
HF_KEY_API = st.secrets["HF_KEY_API"]

# Initialize Pinecone client
pc = PineconeClient(api_key=PINECONE_API_KEY)

# Define index name
index_name = "mindful-chatbot"

# Ensure the index exists
try:
    pc.describe_index(index_name)
except Exception:
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Connect to the existing index
index = pc.Index(index_name)

# Load and process documents (only once per session to avoid reloading)
@st.cache_resource
def load_documents():
    csv_path = "data/qa_dataset.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    document_loader = CSVLoader(csv_path, encoding="utf-8")
    raw_documents = document_loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_documents(raw_documents)

# Load once
docs = load_documents()

# Create or load Pinecone index
@st.cache_resource
def initialize_docsearch():
    try:
        pc.describe_index(index_name)
        return Pinecone.from_existing_index(index_name, HuggingFaceEmbeddings())
    except Exception:
        return Pinecone.from_documents(docs, HuggingFaceEmbeddings(), index_name=index_name)

# Initialize
docsearch = initialize_docsearch()

# Hugging Face Inference Client
inference_client = InferenceClient(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    token=HF_KEY_API,
)

# Prompt template
prompt_template = PromptTemplate(
    template="""
    You are a mental health chatbot. Using the provided information, answer user questions accurately.
    Do not discuss anything unrelated to mental health. If the user expresses distress, suggest seeking professional help.

    Past conversations: {pasts}
    Context: {context}
    Question: {question}
    Answer:
    """,
    input_variables=["context", "question", "pasts"]
)

def call_llm(context, question, pasts):
    full_prompt = prompt_template.format(context=context, question=question, pasts=pasts)
    response = inference_client.text_generation(
        prompt=full_prompt,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        stop=["\n\n"],
    )
    return response.strip()

def generate_response(user_input, chat_history=[]):
    retrieved_docs = docsearch.similarity_search(user_input, k=3)
    context = "\n".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else "No relevant documents found."
    pasts = "\n".join([msg["content"] for msg in chat_history if msg["role"] == "user"])
    return call_llm(context=context, question=user_input, pasts=pasts)
