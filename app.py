import streamlit as st
import os
from pathlib import Path
from query import main as rag_main, get_ollama_models, get_groq_models
from indexer import main as index_main

CODE_DIR = "./code_docs"

st.title("RAG AI Assistant")

with st.expander("How does this work?"):
    try:
        with open("rag_explanation.md", "r", encoding="utf-8") as f:
            st.markdown(f.read())
    except FileNotFoundError:
        st.warning("RAG explanation file not found.")


# Initialisation session_state
if "provider" not in st.session_state:
    st.session_state.provider = "ollama"
if "last_model" not in st.session_state:
    st.session_state.last_model = {"ollama": None, "groq": None}

# --- Provider selection ---
provider = st.selectbox(
    "Choose a model provider",
    ["ollama", "groq"],
    index=["ollama", "groq"].index(st.session_state.provider),
)
if provider != st.session_state.provider:
    st.session_state.provider = provider
    st.rerun()

# --- Model Selection ---
if st.session_state.provider == "ollama":
    models = get_ollama_models()
    if not models:
        st.warning("No local Ollama model found. Start Ollama and add a model (e.g., `ollama run gemma:2b`).")
else:  # "groq"
    models = get_groq_models()
    if not models:
        st.error("Unable to retrieve Groq models. Please check your API key and connection.")

# Keep a separate selected model per provider
model_state_key = f"selected_model_{st.session_state.provider}"

# Initialize if needed
if model_state_key not in st.session_state:
    if models:
        # Pick default
        if st.session_state.provider == "ollama" and "gemma:2b" in models:
            st.session_state[model_state_key] = "gemma:2b"
        elif st.session_state.provider == "groq" and "llama3-8b-8192" in models:
            st.session_state[model_state_key] = "llama3-8b-8192"
        else:
            st.session_state[model_state_key] = models[0]
    else:
        st.session_state[model_state_key] = None

# Selection UI
selected_model = st.selectbox(
    "Choose a model",
    models,
    index=models.index(st.session_state[model_state_key]) if st.session_state[model_state_key] in models else 0,
    key=model_state_key
)

# Save globally for downstream code
st.session_state.selected_model = selected_model


# Create folder if necessary
os.makedirs(CODE_DIR, exist_ok=True)

# File upload
uploaded_files = st.file_uploader(
    "Add files to the knowledge base", accept_multiple_files=True
)
if uploaded_files:
    for f in uploaded_files:
        with open(os.path.join(CODE_DIR, f.name), "wb") as out_file:
            out_file.write(f.getbuffer())
    st.success(f"{len(uploaded_files)} file(s) added.")

    # Automatic indexing
    with st.spinner("Indexing in progress..."):
        index_main(CODE_DIR, index_path="faiss.index", mapping_path="mapping.pkl")
    st.success("Indexing finished.")


# Current files list with delete option
st.subheader("Currently indexed files")
files = sorted(Path(CODE_DIR).glob("*.*"))
if files:
    for f in files:
        col1, col2 = st.columns([4, 1])
        col1.write(f.name)
        if col2.button("🗑 Delete", key=f"del_{f.name}"):
            os.remove(f)
            st.warning(f"{f.name} deleted.")
            with st.spinner("Updating index..."):
                index_main(CODE_DIR, index_path='faiss.index', mapping_path='mapping.pkl')
            st.success("Index updated after deletion")

            # Reload page
            st.write("""
                <script>
                window.location.reload();
                </script>
                """, unsafe_allow_html=True)
else:
    st.write("No files indexed.")

# Question area
question = st.text_area("Ask question about your files...")

if st.button("Submit prompt"):
    if not question.strip():
        st.warning("Submit a prompt.")
    elif not st.session_state.selected_model:
        st.error("No model selected or available.")
    else:
        with st.spinner("Processing..."):
            prompt, answer = rag_main(
                question,
                st.session_state.provider,
                st.session_state.selected_model,
            )

        with st.expander("See the context added to your prompt"):
            st.code(prompt)

        st.subheader("Model's response")
        st.markdown(answer)
