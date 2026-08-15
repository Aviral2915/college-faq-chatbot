import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def get_rag_chain():
    """Loads FAISS vector index and builds a LangChain RAG pipeline."""
    if not os.path.exists("faiss_index"):
        return None

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set. Please set `GROQ_API_KEY=your_key` in a `.env` file.")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2, groq_api_key=api_key)

    prompt = ChatPromptTemplate.from_template("""
You are an AI assistant for BMS College of Engineering (BMSCE).
Answer the student's question accurately using only the context provided.
Always refer to the institution as "BMS College of Engineering" or "BMSCE".

Context:
{context}

Question:
{question}

Answer:
""")

    # LangChain LCEL RAG Chain
    chain = (
        {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), "question": lambda x: x}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

def ask_question(query: str) -> str:
    """Invokes the RAG chain to answer a query."""
    try:
        chain = get_rag_chain()
        if not chain:
            return "[MISSING INDEX] Vector index missing. Please upload documents or run vector_store.py."
        return chain.invoke(query)
    except ValueError as ve:
        return f"[CONFIG ERROR] {str(ve)}"
    except Exception as e:
        return f"[ERROR] {str(e)}"