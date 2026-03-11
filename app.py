import streamlit as st
import os
from dotenv import load_dotenv
from src.data_ingestion import load_and_chunk_financial_reports
from src.vector_store import build_and_save_index, load_local_index
from src.llm_pipeline import get_rag_chain

load_dotenv()

# --- Config ---
DATA_DIR = "data"
INDEX_DIR = "faiss_index"

st.set_page_config(page_title="Corporate Insight Engine", page_icon="🏦", layout="wide")
st.title("🏦 Corporate Insight Engine")
st.markdown("Query SEC filings and earnings transcripts instantly to extract quantitative insights.")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar: Database Management ---
with st.sidebar:
    st.header("Database Management")
    
    if st.button("🏗️ Process Documents & Build Index"):
        if not os.environ.get("OPENAI_API_KEY"):
            st.error("Missing OPENAI_API_KEY in .env file.")
        else:
            with st.spinner("Chunking PDFs..."):
                chunks = load_and_chunk_financial_reports(DATA_DIR)
                if chunks:
                    with st.spinner(f"Embedding {len(chunks)} chunks..."):
                        success = build_and_save_index(chunks, INDEX_DIR)
                        if success:
                            st.session_state.vectorstore = load_local_index(INDEX_DIR)
                            st.success("Database built and loaded successfully!")
                        else:
                            st.error("Failed to build database.")
                else:
                    st.warning(f"No PDFs found in the '{DATA_DIR}' folder.")

    if st.button("📂 Load Existing Index"):
        with st.spinner("Loading database..."):
            vectorstore = load_local_index(INDEX_DIR)
            if vectorstore:
                st.session_state.vectorstore = vectorstore
                st.success("Database loaded! Ready to chat.")
            else:
                st.error("No local index found. Please build it first.")

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- Main Chat Interface ---
st.divider()

# Render chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Ask about Q3 revenue, forward guidance, or risk factors..."):
    if st.session_state.vectorstore is None:
        st.warning("⚠️ Please load or build the database from the sidebar first.")
    else:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing financial context..."):
                try:
                    rag_chain = get_rag_chain(st.session_state.vectorstore)
                    response = rag_chain.invoke({"input": prompt})
                    answer = response["answer"]
                    
                    st.markdown(answer)
                    
                    # Optional: Add an expander to show the specific chunks retrieved
                    with st.expander("View Source Context"):
                        for i, doc in enumerate(response["context"], 1):
                            st.markdown(f"**Source {i}:** `{doc.metadata.get('source', 'Unknown')}`")
                            st.caption(doc.page_content)
                            st.divider()
                            
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"An error occurred during retrieval: {e}")