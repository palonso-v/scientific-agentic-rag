
from pathlib import Path
import json

import numpy as np
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

BASE = Path(__file__).resolve().parent
SOURCE = BASE / "extracted" / "paper.json"
INDEX = BASE / "index"
INDEX.mkdir(exist_ok=True)

EMBEDDING_MODEL = "text-embedding-3-small"
encoding = tiktoken.get_encoding("cl100k_base")


def split_text(text, size=400, overlap=60):
    if not 0 <= overlap < size:
        raise ValueError("Invalid chunk size or overlap.")

    tokens = encoding.encode(text)
    step = size - overlap

    return [
        encoding.decode(tokens[i:i + size])
        for i in range(0, len(tokens), step)
        if tokens[i:i + size]
    ]


with SOURCE.open(encoding="utf-8") as f:
    pages = json.load(f)

chunks = []

for page in pages:
    for j, text in enumerate(
        split_text(page["text"]),
        start=1
    ):
        chunks.append({
            "id": f"paper_p{page['page']}_c{j}",
            "paper_id": "paper",
            "page": page["page"],
            "text": text
        })

if not chunks:
    raise ValueError("No text found to index.")

print(f"Created {len(chunks)} chunks.")

vectors = []

# Batch the requests rather than embedding one chunk at a time.
batch_size = 32

for start in range(0, len(chunks), batch_size):
    batch = chunks[start:start + batch_size]

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[item["text"] for item in batch]
    )

    batch_vectors = [
        item.embedding
        for item in sorted(
            response.data,
            key=lambda item: item.index
        )
    ]

    vectors.extend(batch_vectors)

vectors = np.asarray(vectors, dtype=np.float32)

norms = np.linalg.norm(
    vectors, axis=1, keepdims=True
)

if np.any(norms == 0):
    raise ValueError("Encountered a zero embedding.")

vectors = vectors / norms

assert len(chunks) == len(vectors)

np.save(INDEX / "embeddings.npy", vectors)

with (INDEX / "chunks.json").open(
    "w", encoding="utf-8"
) as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

with (INDEX / "manifest.json").open(
    "w", encoding="utf-8"
) as f:
    json.dump(
        {
            "embedding_model": EMBEDDING_MODEL,
            "number_of_chunks": len(chunks),
            "embedding_dimension": vectors.shape[1],
            "chunk_size": 400,
            "overlap": 60
        },
        f,
        indent=2
    )

print("Embedding matrix:", vectors.shape)
print("Index saved to:", INDEX)
