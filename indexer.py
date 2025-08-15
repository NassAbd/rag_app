import os
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
from pypdf import PdfReader
from docx import Document

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
META_FILE = "index_meta.pkl"
CACHE_FILE = "embedding_cache.pkl"  # Cache embeddings to avoid recalculation


def list_all_files(root_dir):
    """Lists all files in the given directory, non-recursively."""
    return [p for p in Path(root_dir).iterdir() if p.is_file()]


def extract_text_from_file(file_path):
    """Extracts text from a file based on its extension."""
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        try:
            reader = PdfReader(file_path)
            return "".join(page.extract_text() for page in reader.pages)
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")
            return None
    elif ext == ".docx":
        try:
            doc = Document(file_path)
            return "\n".join(para.text for para in doc.paragraphs)
        except Exception as e:
            print(f"Error reading DOCX {file_path}: {e}")
            return None
    # Plain text formats
    elif ext in [".py", ".txt", ".md", ".json"]:
        try:
            return file_path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"Error reading text file {file_path}: {e}")
            return None
    else:
        print(f"Unsupported file type: {file_path.name}")
        return None


def chunk_code(code, max_lines=20):
    lines = code.splitlines()
    chunks = []
    for i in range(0, len(lines), max_lines):
        chunk = "\n".join(lines[i : i + max_lines])
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

    # Load metadata and embedding cache
    meta = load_pickle(META_FILE)
    cache = load_pickle(CACHE_FILE)

    current_files = list_all_files(code_dir)
    current_files_set = set(map(str, current_files))

    changed = False

    # Identify deleted files
    deleted_files = set(meta.keys()) - current_files_set
    if deleted_files:
        changed = True
        print(
            f"🗑 Deletion detected: {len(deleted_files)} file(s) removed from the index"
        )
        for f in deleted_files:
            meta.pop(f, None)
            cache.pop(f, None)

    updated_meta = meta.copy()

    # (Re)generate embeddings for new or modified files
    for f_path in current_files:
        f_str = str(f_path)
        mtime = os.path.getmtime(f_str)
        if f_str not in meta or meta[f_str] < mtime:
            changed = True
            print(f"📄 Indexing: {f_str}")
            content = extract_text_from_file(f_path)

            if content:
                chunks = chunk_code(content)
                embeddings = model.encode(chunks, show_progress_bar=False)
                cache[f_str] = (chunks, embeddings)
                updated_meta[f_str] = mtime
            else:
                # If content could not be extracted, remove from cache/meta if it existed
                if f_str in updated_meta:
                    updated_meta.pop(f_str)
                    cache.pop(f_str, None)
                    print(f"Skipped and removed failed file from index: {f_str}")

    # If nothing has changed, exit directly
    if not changed:
        print("ℹ No new or modified files, index is unchanged.")
        return

    # Full reconstruction of the FAISS index from the present files
    all_chunks = []
    all_embeddings = []
    for f_str in sorted(updated_meta.keys()):
        if f_str in cache:  # Ensure file was processed successfully
            chunks, embeddings = cache[f_str]
            all_chunks.extend(chunks)
            all_embeddings.extend(embeddings)

    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    if all_embeddings:
        import numpy as np

        index.add(np.array(all_embeddings, dtype="float32"))

    # Save
    faiss.write_index(index, index_path)
    save_pickle(all_chunks, mapping_path)
    save_pickle(updated_meta, META_FILE)
    save_pickle(cache, CACHE_FILE)

    print(
        f"✅ Index rebuilt with {len(all_chunks)} chunks from {len(updated_meta)} files."
    )

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python indexer.py <dossier_code_python>")
        exit(1)
    main(sys.argv[1])
