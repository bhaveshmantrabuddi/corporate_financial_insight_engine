from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

def get_rag_chain(vectorstore):
    """Builds the Retrieval-Augmented Generation chain."""
    
    # Fetch top 5 most relevant chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Strict financial prompt to enforce accuracy and citations
    system_prompt = (
        "You are an expert quantitative analyst. Answer the user's question using ONLY the provided context. "
        "Do not hallucinate or use outside knowledge. If the context does not contain the answer, "
        "explicitly state 'The provided documents do not contain this information.' "
        "Always cite the source document filename in your response.\n\n"
        "Context: {context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Temperature 0 is critical for factual financial retrieval
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    qa_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, qa_chain)