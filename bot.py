import os
import streamlit as st
import numpy as np
import faiss
from pypdf import PdfReader
import google.generativeai as genai
from dotenv import load_dotenv

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize session state
if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "index" not in st.session_state:
    st.session_state.index = None

def read_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"
    return text

def chunk_text(text, size=500):
    return [text[i:i+size] for i in range(0, len(text), size)]

def get_embeddings(text):
    embed = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_document"
    )
    return embed["embedding"]

def build_index(chunks):
    embeddings = np.array([get_embeddings(chunk) for chunk in chunks]).astype("float32")
    dim = embeddings.shape[1]
    idx = faiss.IndexFlatL2(dim)
    idx.add(embeddings)
    return idx

def search(query, k=3):
    query_vector = np.array([get_embeddings(query)]).astype("float32")
    D, I = st.session_state.index.search(query_vector, k)
    return [st.session_state.chunks[i] for i in I[0]]

def answer_question(query):
    if not st.session_state.chunks or st.session_state.index is None:
        return "❗ Please upload and process a PDF first."
    context = "\n".join(search(query))
    prompt = f"Use the context below to answer:\n\n{context}\n\nQuestion: {query}\nAnswer:"
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()

# Streamlit UI
st.set_page_config(page_title="PDF Q&A Bot", layout="wide")
st.markdown("<h1 style='text-align: center;'>📘 PDF Q&A Chatbot</h1>", unsafe_allow_html=True)

# Layout - Two Columns
left_col, right_col = st.columns(2)

# === Left Column: Upload & Process ===
with left_col:
    st.markdown("### 📁 Upload & Process PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")
    if st.button("Process PDF"):
        if uploaded_file is not None:
            try:
                text = read_pdf(uploaded_file)
                st.session_state.chunks = chunk_text(text)
                st.session_state.index = build_index(st.session_state.chunks)
                st.success("✅ PDF processed successfully.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("Please upload a PDF file.")

# === Right Column: Q&A ===
with right_col:
    st.markdown("### 💬 Ask a Question")
    query = st.text_input("Enter your question")
    if st.button("Ask"):
        if query.strip():
            with st.spinner("Generating answer..."):
                answer = answer_question(query)
                st.text_area("Answer", value=answer, height=200)
        else:
            st.warning("Please enter a question.")
