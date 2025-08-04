from sentence_transformers import SentenceTransformer
import os
import json
import faiss
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

data_dir = "data"
output_dir = "embeddings"
os.makedirs(output_dir, exist_ok=True)

texts = []
doc_ids = []

# Load all text files
for fname in sorted(os.listdir(data_dir)):
    if fname.endswith(".txt"):
        with open(os.path.join(data_dir, fname), "r", encoding="utf-8") as f:
            content = f.read()
            chunks = [content[i:i+512] for i in range(0, len(content), 512)]
            texts.extend(chunks)
            doc_ids.extend([fname] * len(chunks))

# Generate embeddings
embeddings = model.encode(texts, show_progress_bar=True)

# Save index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings).astype("float32"))
faiss.write_index(index, os.path.join(output_dir, "faiss.index"))

# Save metadata
with open(os.path.join(output_dir, "metadata.json"), "w") as f:
    json.dump({"texts": texts, "doc_ids": doc_ids}, f)

print(f"✅ Embedded {len(texts)} chunks from {len(set(doc_ids))} files.")
