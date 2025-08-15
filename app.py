# app.py
import streamlit as st
import os
from pathlib import Path
from query import main as rag_main, get_ollama_models, get_groq_models
from indexer import main as index_main

CODE_DIR = "./code_docs"

st.title("RAG Python Code Assistant")

# Initialisation session_state
if "provider" not in st.session_state:
    st.session_state.provider = "ollama"
if "last_model" not in st.session_state:
    st.session_state.last_model = {"ollama": None, "groq": None}

# --- Sélection Provider ---
provider = st.selectbox(
    "Choisissez un provider de modèle",
    ["ollama", "groq"],
    index=["ollama", "groq"].index(st.session_state.provider),
)
if provider != st.session_state.provider:
    st.session_state.provider = provider
    st.rerun()  # Forcer le rechargement pour maj la liste des modèles

# --- Sélection Modèle ---
if st.session_state.provider == "ollama":
    models = get_ollama_models()
    if not models:
        st.warning("Aucun modèle Ollama local trouvé. Démarrez Ollama et ajoutez un modèle (ex: `ollama run gemma:2b`).")
else: # "groq"
    models = get_groq_models()
    if not models:
        st.error("Impossible de récupérer les modèles Groq. Vérifiez votre clé API et la connexion.")

model_key = f"model_{st.session_state.provider}"
last_used_model = st.session_state.last_model.get(st.session_state.provider)

# Déterminer l'index du modèle par défaut
default_model_index = 0
if last_used_model and last_used_model in models:
    default_model_index = models.index(last_used_model)
elif models:
    # Si aucun dernier modèle utilisé, prendre une valeur par défaut intelligente
    if st.session_state.provider == "ollama" and "gemma:2b" in models:
        default_model_index = models.index("gemma:2b")
    elif st.session_state.provider == "groq" and "llama3-8b-8192" in models:
        default_model_index = models.index("llama3-8b-8192")

if "selected_model" not in st.session_state:
    st.session_state.selected_model = models[default_model_index] if models else None

selected_model = st.selectbox(
    "Choisissez un modèle",
    models,
    index=models.index(st.session_state.selected_model) if st.session_state.selected_model in models else default_model_index,
    key=model_key,
)

# Mettre à jour le dernier modèle utilisé et le modèle sélectionné
st.session_state.last_model[st.session_state.provider] = selected_model
st.session_state.selected_model = selected_model


# Création dossier si inexistant
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
    st.success("Indexing finished ✅")


# Current files list with delete option
st.subheader("Currently indexed files")
files = sorted(Path(CODE_DIR).glob("*.*"))
if files:
    for f in files:
        col1, col2 = st.columns([4, 1])
        col1.write(f.name)
        if col2.button("🗑 Supprimer", key=f"del_{f.name}"):
            os.remove(f)
            st.warning(f"{f.name} supprimé.")
            with st.spinner("Mise à jour de l'index..."):
                index_main(CODE_DIR, index_path='faiss.index', mapping_path='mapping.pkl')
            st.success("Index mis à jour après suppression ✅")

            # Reload page (contournement si experimental_rerun indisponible)
            st.write("""
                <script>
                window.location.reload();
                </script>
                """, unsafe_allow_html=True)
else:
    st.write("Aucun fichier indexé.")

# 🔹 Zone de question
question = st.text_area("Posez votre question sur le code")

if st.button("Chercher réponse"):
    if not question.strip():
        st.warning("Entrez une question.")
    elif not st.session_state.selected_model:
        st.error("Aucun modèle n'est sélectionné ou disponible.")
    else:
        with st.spinner("Recherche en cours..."):
            prompt, answer = rag_main(
                question,
                st.session_state.provider,
                st.session_state.selected_model,
            )

        with st.expander("Voir le prompt envoyé au modèle"):
            st.code(prompt)

        st.subheader("Réponse du modèle")
        st.markdown(answer)  # Streamlit gère le markdown
