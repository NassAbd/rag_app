# rag_query.py
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import requests
import json
import os

MODEL_NAME = 'all-MiniLM-L6-v2'
EMBEDDING_DIM = 384
API_KEY = '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'

def load_index(index_path='faiss.index', mapping_path='mapping.pkl'):
    if not os.path.exists(index_path) or not os.path.exists(mapping_path):
        print("⚠️ Index ou mapping non trouvé.")
        return None, []

    index = faiss.read_index(index_path)
    with open(mapping_path, 'rb') as f:
        texts = pickle.load(f)
    return index, texts

def embed_query(query, model):
    return model.encode([query])

def search_context(query, index, texts, model, top_k=5):
    if index is None or not texts:
        return "Aucun document indexé."

    q_vec = embed_query(query, model)
    D, I = index.search(q_vec, top_k)

    if not I.any() or len(I[0]) == 0:
        return "Aucun document pertinent trouvé."

    # Sécurité si index retourné est vide ou dépasse la liste texts
    filtered_indices = [i for i in I[0] if i < len(texts)]
    if not filtered_indices:
        return "Aucun document pertinent trouvé."

    return "\n\n".join([texts[i] for i in filtered_indices])

def call_ollama(prompt, model_name="gemma3:1b"):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt
    }
    try:
        response = requests.post(url, json=payload, stream=True)
        response.raise_for_status()

        output = ""
        for line in response.iter_lines():
            if line:
                try:
                    json_data = json.loads(line.decode('utf-8'))
                    output += json_data.get("response", "")
                except json.JSONDecodeError:
                    pass
        return output.strip()
    except Exception as e:
        print(f"Erreur Ollama API: {e}")
        return ""

def call_groq_api(prompt, api_key=API_KEY):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    json_data = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 300,
        "temperature": 0.3
    }
    try:
        response = requests.post(url, headers=headers, json=json_data)
        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['message']['content']
    except Exception as e:
        print(f"Erreur Groq API: {e}")
        return ""

def main(question, provider='ollama'):
    model = SentenceTransformer(MODEL_NAME)
    index, texts = load_index()

    context = search_context(question, index, texts, model)
    prompt = f"Contexte code:\n{context}\n\nQuestion:\n{question}\n\nRéponse :"

    if provider == 'ollama':
        answer = call_ollama(prompt)
    else:
        answer = call_groq_api(prompt)

    if not answer:
        answer = "⚠️ Aucun retour du modèle."

    return prompt, answer

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python rag_query.py <question> [provider: ollama/groq]")
        exit(1)
    question = sys.argv[1]
    provider = sys.argv[2] if len(sys.argv) > 2 else 'ollama'
    prompt, answer = main(question, provider)
    print("=== Prompt ===")
    print(prompt)
    print("\n=== Réponse ===")
    print(answer)
