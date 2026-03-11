# 🏦 Corporate Insight Engine (Financial RAG Pipeline)

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3.x-green?logo=chainlink&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-API-black?logo=openai&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-red?logo=streamlit&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Store-blueviolet)

An enterprise-structured Retrieval-Augmented Generation (RAG) pipeline designed to extract highly accurate quantitative insights from dense, unstructured SEC filings (10-Ks) and corporate earnings transcripts.



## 🧠 Architectural Highlights

### 1. Decoupled Offline Indexing
To optimize runtime latency and reduce API costs, the architecture separates data ingestion from the user interface. 
* **Offline Phase:** `src/data_ingestion.py` and `src/vector_store.py` handle the heavy lifting of parsing PDFs, applying semantic chunking (`RecursiveCharacterTextSplitter`), and generating embeddings via `text-embedding-3-small`. The resulting vector database is saved locally via FAISS.
* **Online Phase:** The Streamlit frontend (`app.py`) simply loads the pre-computed FAISS index into memory, resulting in instant application startup and lightning-fast retrieval.

### 2. Hallucination Prevention & Prompt Engineering
In financial analytics, LLM hallucinations are unacceptable. This engine enforces strict grounding through:
* **Zero-Temperature Generation:** The `gpt-4o-mini` model is configured with `temperature=0` to ensure deterministic, highly factual outputs.
* **Source Citations:** The system prompt explicitly restricts the LLM from using outside knowledge and mandates that every response cites the specific source document filename retrieved from the vector metadata.

### 3. Financial Context Preservation
Financial documents contain complex, interdependent clauses. The chunking strategy utilizes a 1000-character size with a 200-character overlap, ensuring that trailing financial caveats or risk factors are not artificially severed from their primary statements during vectorization.

## 🛠️ Tech Stack
* **Frontend:** Streamlit
* **Orchestration:** LangChain
* **Embeddings:** OpenAI `text-embedding-3-small`
* **LLM Synthesis:** OpenAI `gpt-4o-mini`
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Document Processing:** PyPDF
