
from pathlib import Path
import json

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from shared.openai_client import client

load_dotenv()

INDEX = Path(__file__).resolve().parent / "index_v2"


def load_collection():
    matrices = []
    all_chunks = []
    embedding_model = None

    if not INDEX.is_dir():
        raise FileNotFoundError(
            f"Document index not found: {INDEX}. "
            "Build or provide an index before running retrieval."
        )

    for paper_dir in sorted(INDEX.iterdir()):
        if not paper_dir.is_dir():
            continue

        metadata_path = paper_dir / "metadata.json"
        vectors_path = paper_dir / "embeddings.npy"

        if not metadata_path.exists() or not vectors_path.exists():
            continue

        with metadata_path.open(encoding="utf-8") as f:
            metadata = json.load(f)

        vectors = np.load(
            vectors_path,
            allow_pickle=False
        )

        if len(vectors) != len(metadata["chunks"]):
            raise ValueError(
                f"Corrupted index: {paper_dir.name}"
            )

        model = metadata["embedding_model"]

        if embedding_model is None:
            embedding_model = model
        elif model != embedding_model:
            raise ValueError(
                "The collection contains incompatible "
                "embedding models."
            )

        matrices.append(vectors)
        all_chunks.extend(metadata["chunks"])

    if not matrices:
        raise ValueError("No indexed papers found.")

    matrix = np.vstack(matrices)

    return matrix, all_chunks, embedding_model


vectors = None
chunks = []
model = None


def ensure_collection_loaded():
    global vectors, chunks, model

    if vectors is None:
        vectors, chunks, model = load_collection()

    return vectors, chunks, model


def search(question, k=5):
    vectors, chunks, model = ensure_collection_loaded()

    response = client.embeddings.create(
        model=model,
        input=[question]
    )

    query = np.asarray(
        response.data[0].embedding,
        dtype=np.float32
    )

    if query.shape[0] != vectors.shape[1]:
        raise ValueError("Embedding dimension mismatch.")

    norm = np.linalg.norm(query)

    if norm == 0:
        raise ValueError("Zero query embedding.")

    query /= norm

    scores = vectors @ query
    top_indices = np.argsort(scores)[::-1][:k]

    return [
        {
            **chunks[int(i)],
            "score": float(scores[i])
        }
        for i in top_indices
    ]


if __name__ == "__main__":
    while True:
        question = input("\nQuestion: ").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        for result in search(question):
            print(
                f"\n{result['paper']} "
                f"| Page {result['page']} "
                f"| Score {result['score']:.4f}"
            )
            print(result["text"][:1000])
