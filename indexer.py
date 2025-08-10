import os
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss

MODEL_NAME = 'all-MiniLM-L6-v2'
EMBEDDING_DIM = 384
META_FILE = 'index_meta.pkl'
CACHE_FILE = 'embedding_cache.pkl'  # On garde les embeddings pour éviter recalcul

def list_py_files(root_dir):
    return list(Path(root_dir).rglob("*.py"))

def chunk_code(code, max_lines=20):
    lines = code.splitlines()
    chunks = []
    for i in range(0, len(lines), max_lines):
        chunk = "\n".join(lines[i:i+max_lines])
        chunks.append(chunk)
    return chunks

def load_pickle(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return {}

def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def main(code_dir, index_path='faiss.index', mapping_path='mapping.pkl'):
    model = SentenceTransformer(MODEL_NAME)

    # Charger les métadonnées et cache d'embeddings
    meta = load_pickle(META_FILE)
    cache = load_pickle(CACHE_FILE)

    current_files = list_py_files(code_dir)
    current_files_set = set(map(str, current_files))

    changed = False  # 🔹 Suivi des modifications

    # 🔹 Identifier fichiers supprimés
    deleted_files = set(meta.keys()) - current_files_set
    if deleted_files:
        changed = True
        print(f"🗑 Suppression détectée : {len(deleted_files)} fichier(s) supprimé(s) de l'index")
        for f in deleted_files:
            meta.pop(f, None)
            cache.pop(f, None)

    updated_meta = meta.copy()

    # 🔹 (Re)génération embeddings pour fichiers nouveaux ou modifiés
    for f in current_files:
        f_str = str(f)
        mtime = os.path.getmtime(f_str)
        if f_str not in meta or meta[f_str] < mtime:
            changed = True
            print(f"📄 Indexation : {f_str}")
            code = Path(f_str).read_text(encoding='utf-8')
            chunks = chunk_code(code)
            embeddings = model.encode(chunks, show_progress_bar=False)
            cache[f_str] = (chunks, embeddings)
            updated_meta[f_str] = mtime

    # 🔹 Si rien n’a changé → on sort directement
    if not changed:
        print("ℹ Aucun fichier nouveau ou modifié, index inchangé.")
        return

    # 🔹 Reconstruction complète de l’index FAISS à partir des fichiers présents
    all_chunks = []
    all_embeddings = []
    for f_str in sorted(updated_meta.keys()):
        chunks, embeddings = cache[f_str]
        all_chunks.extend(chunks)
        all_embeddings.extend(embeddings)

    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    if all_embeddings:
        import numpy as np
        index.add(np.array(all_embeddings, dtype='float32'))

    # 🔹 Sauvegarde
    faiss.write_index(index, index_path)
    save_pickle(all_chunks, mapping_path)
    save_pickle(updated_meta, META_FILE)
    save_pickle(cache, CACHE_FILE)

    print(f"✅ Index reconstruit avec {len(all_chunks)} chunks provenant de {len(updated_meta)} fichiers.")

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python indexer.py <dossier_code_python>")
        exit(1)
    main(sys.argv[1])
