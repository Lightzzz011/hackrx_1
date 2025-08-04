from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

# Load model and data
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("embeddings/faiss.index")

with open("embeddings/metadata.json", "r") as f:
    metadata = json.load(f)

texts = metadata["texts"]
doc_ids = metadata["doc_ids"]

def search(query, top_k=5):
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec).astype("float32"), top_k)

    results = []
    for idx in I[0]:
        results.append({
            "doc": doc_ids[idx],
            "text": texts[idx][:500] + "..." if len(texts[idx]) > 500 else texts[idx]
        })

    return results

# --- Test the search ---
if __name__ == "__main__":
    while True:
        query = input("\n❓ Ask a question (or type 'exit'): ")
        if query.lower() == "exit":
            break
        results = search(query)
        for i, res in enumerate(results, 1):
            print(f"\n🔹 Result {i} from {res['doc']}:\n{res['text']}\n")
