import os
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def build_vector_store(data_dir="data", db_path="faiss_index"):
    """Loads PDF/TXT documents, chunks text, embeds, and saves a FAISS vector index."""
    documents = []
    
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            file_path = os.path.join(data_dir, file)
            if file.endswith(".pdf"):
                loader = PyMuPDFLoader(file_path)
                documents.extend(loader.load())
            elif file.endswith(".txt"):
                loader = TextLoader(file_path, encoding="utf-8")
                documents.extend(loader.load())

    if not documents:
        print("[WARNING] No documents found to index.")
        return None

    # Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    # Embeddings and FAISS store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local(db_path)

    print(f"[SUCCESS] Created vector store with {len(chunks)} chunks from {len(documents)} documents.")
    return vector_db

if __name__ == "__main__":
    build_vector_store()