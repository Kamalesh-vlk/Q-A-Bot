import os
import gradio as gr
import numpy as np
import faiss
from pypdf import PdfReader
import google.generativeai as genai
from dotenv import load_dotenv

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Globals
chunks = []
index = None

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
    D, I = index.search(query_vector, k)
    return [chunks[i] for i in I[0]]

def answer_question(query):
    if not chunks or index is None:
        return "Please upload and process a PDF first."
    context = "\n".join(search(query))
    prompt = f"Use the context below to answer:\n\n{context}\n\nQuestion: {query}\nAnswer:"
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()

def process_pdf(file):
    global chunks, index
    try:
        text = read_pdf(file)
        chunks = chunk_text(text)
        index = build_index(chunks)
        return "✅ PDF processed successfully."
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Gradio Interface
with gr.Blocks(css="""
body {
    background-color: #f0f0f0;
}
#container {
    display: flex;
    gap: 20px;
}
#left, #right {
    background: white;
    padding: 20px;
    border-radius: 10px;
    flex: 1;
    box-shadow: 0 0 10px rgba(0,0,0,0.05);
}
""") as demo:

    gr.Markdown("<h2 style='text-align: center;'>📘 Q & A Bot </h2>")

    with gr.Row(elem_id="container"):
        with gr.Column(elem_id="left"):
            gr.Markdown("### 📁 Upload & Process PDF")
            pdf_file = gr.File(label="Upload PDF", file_types=[".pdf"])
            process_button = gr.Button("Process PDF")
            status = gr.Markdown("")

        with gr.Column(elem_id="right"):
            gr.Markdown("### 💬 Ask a Question")
            question = gr.Textbox(label="Your Question")
            ask_button = gr.Button("Ask")
            answer = gr.Textbox(label="Answer", lines=6, interactive=False)

    process_button.click(fn=process_pdf, inputs=pdf_file, outputs=status)
    ask_button.click(fn=answer_question, inputs=question, outputs=answer)

if __name__ == "__main__":
    demo.launch()
