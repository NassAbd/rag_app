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
else:  # "groq"
    models = get_groq_models()
    if not models:
        st.error("Impossible de récupérer les modèles Groq. Vérifiez votre clé API et la connexion.")

# Clé unique pour stocker le modèle choisi par provider
model_key = f"selected_model_{st.session_state.provider}"

# Initialisation une seule fois pour chaque provider
if model_key not in st.session_state:
    if models:
        # Choix par défaut intelligent si aucun historique
        if st.session_state.provider == "ollama" and "gemma:2b" in models:
            st.session_state[model_key] = "gemma:2b"
        elif st.session_state.provider == "groq" and "llama3-8b-8192" in models:
            st.session_state[model_key] = "llama3-8b-8192"
        else:
            st.session_state[model_key] = models[0]
    else:
        st.session_state[model_key] = None

# Sélecteur lié uniquement au provider actuel
selected_model = st.selectbox(
    "Choisissez un modèle",
    models,
    index=models.index(st.session_state[model_key]) if st.session_state[model_key] in models else 0,
    key=model_key,
)

# Mise à jour du dernier modèle utilisé pour ce provider
st.session_state.last_model[st.session_state.provider] = st.session_state[model_key]
st.session_state.selected_model = selected_model


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
