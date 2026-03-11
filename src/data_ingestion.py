import os
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger("financial_rag")

def load_and_chunk_financial_reports(data_dir="data"):
    """Loads PDFs, splits them, and injects source metadata."""
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        logger.warning(f"No documents found in {data_dir}.")
        return []

    documents = []
    for file in os.listdir(data_dir):
        if file.endswith(".pdf"):
            file_path = os.path.join(data_dir, file)
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                
                # Clean up metadata to just show the filename (e.g., 'PLTR_10K.pdf')
                for doc in docs:
                    doc.metadata["source"] = file
                documents.extend(docs)
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")

    # Chunking strategy tailored for dense financial text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    return text_splitter.split_documents(documents)