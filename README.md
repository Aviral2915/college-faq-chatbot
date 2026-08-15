# 🎓 BMSCE FAQ Chatbot - LangChain RAG & PyMuPDF

A minimal, high-performance RAG (Retrieval-Augmented Generation) chatbot for BMS College of Engineering (BMSCE) built with **LangChain**, **PyMuPDF**, **FAISS**, **HuggingFace Embeddings**, **ChatGroq (Llama-3.1)**, and **Streamlit**.

---

## 🌟 Key Features

- **LangChain Expression Language (LCEL)**: Clean, modular, and maintainable RAG pipeline.
- **Multi-Format Document Ingestion**: Load PDF and TXT documents using `PyMuPDFLoader` and `TextLoader`.
- **Text Chunking**: Optimized splitting using `RecursiveCharacterTextSplitter`.
- **FAISS Vector Store**: Fast similarity search using `sentence-transformers/all-MiniLM-L6-v2` embeddings.
- **Interactive Knowledge Base**: Upload custom PDF/TXT documents via Streamlit sidebar and rebuild vector store on the fly.
- **Ultra-Fast LLM Inference**: Powered by Groq's `llama-3.1-8b-instant` model.

---

## 📁 Repository Structure

```
├── app.py              # Streamlit UI with chat interface & sidebar uploader
├── chatbot.py          # LangChain LCEL RAG chain & Groq LLM logic
├── vector_store.py     # Document loader, chunking & FAISS index generator
├── data/               # Document folder for PDFs and TXT files
│   └── data.txt        # Initial FAQ knowledge base
├── faiss_index/        # Generated FAISS vector index & metadata
├── requirements.txt    # Python dependencies
└── .env.example        # Environment variable template
```

---

## 🚀 Quick Start Guide

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/Aviral2915/college-faq-chatbot.git
cd college-faq-chatbot
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
```
*(Get a free API key at [Groq Console](https://console.groq.com))*

### 3. Build Vector Store Index
```bash
python vector_store.py
```

### 4. Launch Streamlit Application
```bash
streamlit run app.py
```

---

## 🛠️ Built With
- **LangChain**
- **PyMuPDF (fitz)**
- **FAISS**
- **Groq API (Llama 3.1 8B)**
- **Sentence-Transformers**
- **Streamlit**
