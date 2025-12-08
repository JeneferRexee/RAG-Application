import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------
# AUTO-DETECT LATEST JSON FILE IN OCR FOLDER
# ------------------------------------------------------------

def get_latest_json_from_ocr(folder="ocr"):
    if not os.path.exists(folder):
        raise FileNotFoundError(f"'ocr' folder not found at: {folder}")

    json_files = [f for f in os.listdir(folder) if f.endswith(".json")]

    if not json_files:
        raise FileNotFoundError("No JSON files found in the 'ocr' folder.")

    # Sort by last modified time
    json_files.sort(key=lambda f: os.path.getmtime(os.path.join(folder, f)), reverse=True)

    latest_file = os.path.join(folder, json_files[0])
    print(f" JSON detected: {latest_file}")
    return latest_file


# ------------------------------------------------------------
# 1. LOAD JSON 
# ------------------------------------------------------------

def load_json_from_team1(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    if "query" not in data or "pages" not in data:
        raise ValueError("JSON must contain 'query' and 'pages' keys.")

    pages = data["pages"]
    query = data["query"]

    return pages, query


# ------------------------------------------------------------
# 2. CHUNKING
# ------------------------------------------------------------

def recursive_chunk(text, separators, max_len):
    text = text.strip()
    if len(text) <= max_len or not separators:
        return [text]

    sep = separators[0]
    parts = text.split(sep)
    chunks = []

    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(part) > max_len:
            chunks.extend(recursive_chunk(part, separators[1:], max_len))
        else:
            chunks.append(part)

    return chunks


def chunk_text(json_pages, max_len=400, separators=None):
    if separators is None:
        separators = ["\n\n", ".", "\n"]

    all_chunks = []

    for entry in json_pages:
        page_num = entry["page"]
        page_text = entry["para"]

        page_chunks = recursive_chunk(page_text, separators, max_len)

        for c in page_chunks:
            all_chunks.append({"text": c, "page": page_num})

    return all_chunks


# ------------------------------------------------------------
# 3. EMBEDDINGS
# ------------------------------------------------------------

def generate_embeddings(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [c["text"] for c in chunks]

    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return embeddings, model


# ------------------------------------------------------------
# 4. BUILD FAISS IN MEMORY
# ------------------------------------------------------------

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index


# ------------------------------------------------------------
# 5. RETRIEVE
# ------------------------------------------------------------

def retrieve_relevant_chunks(query, model, index, chunks, k=5):
    query_vec = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, k)
    return [chunks[i] for i in indices[0]]


# ------------------------------------------------------------
# 6. SAVE ONLY RETRIEVED CHUNKS JSON
# ------------------------------------------------------------

def save_retrieved_json(query, retrieved_chunks, output_path="retrieved_output.json"):
    data = {
        "query": query,
        "retrieved_chunks": retrieved_chunks
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Retrieved chunks saved to {output_path}")
    return output_path


# ------------------------------------------------------------
# 7. MAIN PIPELINE
# ------------------------------------------------------------

def process_rag_pipeline(output_path="retrieved_output.json", k=5):

    json_path = get_latest_json_from_ocr("ocr")

    pages, query = load_json_from_team1(json_path)

    chunks = chunk_text(pages)

    embeddings, model = generate_embeddings(chunks)

    index = build_faiss_index(embeddings)

    retrieved_chunks = retrieve_relevant_chunks(query, model, index, chunks, k)

    save_retrieved_json(query, retrieved_chunks, output_path)

    return retrieved_chunks


# ------------------------------------------------------------
# RUN - Main
# ------------------------------------------------------------

if __name__ == "__main__":
    process_rag_pipeline()