import os
import logging
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()
logger = logging.getLogger("financial_rag")

def build_and_save_index(chunks, save_path="faiss_index"):
    """Embeds document chunks and saves the FAISS index locally."""
    if not chunks:
        return False

    try:
        # Using the newer, cheaper, and better embedding model
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        vectorstore = FAISS.from_documents(chunks, embeddings)
        
        os.makedirs(save_path, exist_ok=True)
        vectorstore.save_local(save_path)
        return True
    except Exception as e:
        logger.error(f"Failed to build vector store: {e}")
        return False

def load_local_index(load_path="faiss_index"):
    """Loads a pre-computed FAISS index from disk."""
    try:
        if not os.path.exists(load_path):
            return None
            
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        return FAISS.load_local(
            load_path, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        logger.error(f"Failed to load vector store: {e}")
        return None