# app.py
import streamlit as st
from streamlit.runtime.scriptrunner import RerunException
import os
from pathlib import Path
from query import main as rag_main
from indexer import main as index_main

CODE_DIR = './code_docs'

st.title("RAG Python Code Assistant")

if 'provider' not in st.session_state:
    st.session_state.provider = 'ollama'

# Choix provider
provider = st.selectbox("Choisissez un provider de modèle", ['ollama', 'groq'])
st.session_state.provider = provider

# Création dossier si inexistant
os.makedirs(CODE_DIR, exist_ok=True)

# 🔹 Upload de fichiers
uploaded_files = st.file_uploader("Ajouter des fichiers Python (.py)", type='py', accept_multiple_files=True)
if uploaded_files:
    for f in uploaded_files:
        with open(os.path.join(CODE_DIR, f.name), "wb") as out_file:
            out_file.write(f.getbuffer())
    st.success(f"{len(uploaded_files)} fichier(s) ajouté(s).")

    # Indexation automatique
    with st.spinner("Indexation en cours..."):
        index_main(CODE_DIR, index_path='faiss.index', mapping_path='mapping.pkl')
    st.success("Indexation terminée ✅")

# 🔹 Liste des fichiers actuels avec option suppression
st.subheader("Fichiers actuellement indexés")
files = sorted(Path(CODE_DIR).glob("*.py"))
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
    else:
        with st.spinner("Recherche en cours..."):
            prompt, answer = rag_main(question, st.session_state.provider)

        with st.expander("Voir le prompt envoyé au modèle"):
            st.code(prompt)

        st.subheader("Réponse du modèle")
        st.markdown(answer)  # Streamlit gère le markdown
