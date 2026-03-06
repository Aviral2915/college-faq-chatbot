import os
import pickle
import faiss
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in your .env file or deployment environment.")

# Initialize Groq client
client = Groq(api_key=api_key)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index
index = faiss.read_index("faiss_index/index.faiss")

# Load stored texts
with open("faiss_index/texts.pkl", "rb") as f:
    texts = pickle.load(f)


def ask_question(query):

    # Convert query to embedding
    query_vector = model.encode([query])

    # Search similar documents
    distances, indices = index.search(query_vector, k=2)

    # Retrieve context
    context = ""
    for i in indices[0]:
        if i < len(texts):
            context += texts[i] + "\n"

    # Create prompt
    prompt = f"""
You are an AI assistant for BMS College of Engineering (BMSCE).

Use the provided context if relevant to answer the question. Always refer to the institution as "BMS College of Engineering" or "BMSCE" - never as Bangalore Medical College or any other institution.

Context:
{context}

Question:
{query}

Answer clearly for a student, ensuring all references to the college are accurate.
"""

    # Call Groq API
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model="llama-3.1-8b-instant"
        )
        # Return AI response
        return response.choices[0].message.content
    except Exception as e:
        if "authentication" in str(e).lower() or "unauthorized" in str(e).lower():
            return "⚠️ API Authentication Error: Please check your GROQ_API_KEY configuration."
        else:
            return f"⚠️ API Error: {str(e)}"