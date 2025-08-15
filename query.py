import faiss
import pickle
from sentence_transformers import SentenceTransformer
import requests
import os
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = 'all-MiniLM-L6-v2'
EMBEDDING_DIM = 384
API_KEY = os.getenv("GROQ_API_KEY")

def load_index(index_path='faiss.index', mapping_path='mapping.pkl'):
    if not os.path.exists(index_path) or not os.path.exists(mapping_path):
        print("Index or mapping not found.")
        return None, []

    index = faiss.read_index(index_path)
    with open(mapping_path, 'rb') as f:
        texts = pickle.load(f)
    return index, texts

def embed_query(query, model):
    return model.encode([query])

def search_context(query, index, texts, model, top_k=5):
    if index is None or not texts:
        return "No documents indexed."

    q_vec = embed_query(query, model)
    D, I = index.search(q_vec, top_k)

    if not I.any() or len(I[0]) == 0:
        return "No relevant documents found."

    # Sécurité si index retourné est vide ou dépasse la liste texts
    filtered_indices = [i for i in I[0] if i < len(texts)]
    if not filtered_indices:
        return "No relevant documents found."

    return "\n\n".join([texts[i] for i in filtered_indices])

def get_ollama_models():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        response.raise_for_status()
        models = response.json()["models"]
        return [m["name"] for m in models]
    except Exception as e:
        print(f"Erreur API Ollama (tags): {e}")
        return []

def get_groq_models(api_key=API_KEY):
    url = "https://api.groq.com/openai/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()['data']
        return [m['id'] for m in data]
    except Exception as e:
        print(f"Groq API error (models): {e}")
        return []


def call_ollama(prompt, model_name="gemma:2b"):
    url = "http://localhost:11434/api/generate"
    payload = {"model": model_name, "prompt": prompt, "stream": False}
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        print(f"Ollama API error: {e}")
        return ""


def call_groq_api(prompt, model_name="llama3-8b-8192", api_key=API_KEY):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    json_data = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 300,
        "temperature": 0.3,
    }
    try:
        response = requests.post(url, headers=headers, json=json_data)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Groq API error: {e}")
        return ""


def main(question, provider="ollama", model_name=None):
    model = SentenceTransformer(MODEL_NAME)
    index, texts = load_index()

    context = search_context(question, index, texts, model)
    prompt = f"Context code:\n{context}\n\nQuestion:\n{question}\n\nResponse :"

    if provider == "ollama":
        answer = call_ollama(prompt, model_name=model_name)
    else:
        answer = call_groq_api(prompt, model_name=model_name)

    if not answer:
        answer = "No response from model."

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
    print("\n=== Response ===")
    print(answer)
