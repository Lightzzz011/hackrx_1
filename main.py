import os
import re
import json
import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import openai
load_dotenv()
openai_api_key = os.getenv("key")
client = openai.OpenAI(api_key=openai_api_key)

model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("embeddings/faiss.index")

with open("embeddings/metadata.json", "r") as f:
    metadata = json.load(f)

texts = metadata["texts"]
doc_ids = metadata["doc_ids"]


def parse_query(query):
    age = re.search(r"\d{2}", query)
    gender = "male" if "male" in query.lower() or "m" in query.lower() else "female"
    procedure = re.findall(r"[a-zA-Z ]+surgery", query.lower())
    location = re.search(r"in ([a-zA-Z]+)", query.lower())
    months = re.search(r"(\d+)[-\s]?month", query)

    return {
        "age": int(age.group()) if age else None,
        "gender": gender,
        "procedure": procedure[0].strip() if procedure else "unknown",
        "location": location.group(1) if location else "unknown",
        "policy_duration_months": int(months.group(1)) if months else None
    }

# ------------------------------
# Semantic search with FAISS
# ------------------------------
def search(query, top_k=5):
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec).astype("float32"), top_k)
    return [texts[idx] for idx in I[0]]

# ------------------------------
# LLM-based policy decision
# ------------------------------
def evaluate_with_llm(parsed, relevant_chunks):
    prompt = f"""
User Query Details:
{json.dumps(parsed, indent=2)}

Relevant Policy Clauses:
{"".join(f"\nClause {i+1}: {chunk}" for i, chunk in enumerate(relevant_chunks))}

Based on the above, answer the following in JSON format:
{{
  "decision": "Approved / Rejected",
  "amount": "number or 0",
  "justification": [
    "short explanation with clause references"
  ]
}}
"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

# ------------------------------
# Run interactive session
# ------------------------------
if __name__ == "__main__":
    while True:
        user_query = input("\n💬 Enter a query (or 'exit'): ")
        if user_query.lower() == "exit":
            break

        parsed = parse_query(user_query)
        relevant_chunks = search(user_query)
        result = evaluate_with_llm(parsed, relevant_chunks)

        print("\n🧾 Final Decision:\n", result)
