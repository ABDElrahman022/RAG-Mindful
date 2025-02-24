# MongoDB Connection & Chat Storage

from pymongo import MongoClient
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Connect to MongoDB
client = MongoClient(os.getenv("MONGO_URI"))
db = client["mindful_chatbot"]
chat_collection = db["chats"]

def save_chat(session_id, role, content):
    """Save a chat message to MongoDB."""
    chat_collection.insert_one({
        "session_id": session_id,
        "role": role,
        "content": content
    })

def get_chat_history(session_id):
    """Retrieve chat history for a given session ID."""
    return list(chat_collection.find({"session_id": session_id}, {"_id": 0, "role": 1, "content": 1}))
