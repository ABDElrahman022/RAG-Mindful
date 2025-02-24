from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
from pinecone import ServerlessSpec
from pinecone import Pinecone as PineconeClient

# Load environment variables
load_dotenv()

# Load API keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HF_KEY_API = os.getenv("HF_KEY_API")

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
        dimension=768,  # Embedding model's dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Connect to the existing index
index = pc.Index(index_name)

# Load and process documents
csv_path = "data/qa_dataset.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"File not found: {csv_path}")

document_loader = CSVLoader(csv_path, encoding="utf-8")  
raw_documents = document_loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(raw_documents)

# Create embeddings and store them in Pinecone
embeddings = HuggingFaceEmbeddings()

# Initialize or load Pinecone index
try:
    pc.describe_index(index_name)
    docsearch = Pinecone.from_existing_index(index_name, embeddings)
except Exception:
    docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)

# Define the LLM (Hugging Face Mixtral-8x7B)
llm = HuggingFaceHub(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    model_kwargs={"temperature": 0.7, "top_p": 0.9, "top_k": 50},
    huggingfacehub_api_token=HF_KEY_API
    )

# Define the prompt template for RAG
template = """
    You are a mental health chatbot. Using the provided information, answer user questions accurately.
    Do not discuss anything unrelated to mental health. If the user expresses distress, suggest seeking professional help.

    Past conversations: {pasts}
    Context: {context}
    Question: {question}
    Answer:
"""
prompt = PromptTemplate(template=template, input_variables=["context", "question", "pasts"])

# Define the RAG pipeline
rag_chain = prompt | llm

import re

def generate_response(user_input, chat_history=[]):
    """Generate a chatbot response using RAG and previous chat history."""
    retrieved_docs = docsearch.similarity_search(user_input, k=3)
    context = "\n".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else "No relevant documents found."

    result = rag_chain.invoke({
        "context": context,
        "question": user_input,
        "pasts": chat_history if isinstance(chat_history, list) else [chat_history]
    })

    # 🔹 If `result` is a dictionary containing an "answer" key, return its value
    if isinstance(result, dict) and "answer" in result:
        return result["answer"].strip()  # Return only the assistant's response
    
    # 🔹 If `result` is a string containing "Answer:", extract the response using Regex
    match = re.search(r"Answer:\s*(.*)", str(result), re.DOTALL)
    if match:
        return match.group(1).strip()

    # 🔹 If no answer is found, return the cleaned result
    return str(result).strip()
