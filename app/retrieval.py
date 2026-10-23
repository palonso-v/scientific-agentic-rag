from pathlib import Path
import json
import re

import numpy as np
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

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
            allow_pickle=False,
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
bm25_index = None


def ensure_collection_loaded():
    global vectors, chunks, model

    if vectors is None:
        vectors, chunks, model = load_collection()

    return vectors, chunks, model


def tokenize_for_bm25(text: str) -> list[str]:
    return re.findall(
        r"\b\w+\b",
        text.lower(),
    )


def ensure_bm25_loaded():
    global bm25_index

    _, chunks, _ = ensure_collection_loaded()

    if bm25_index is None:
        tokenized_corpus = [
            tokenize_for_bm25(chunk["text"])
            for chunk in chunks
        ]

        bm25_index = BM25Okapi(
            tokenized_corpus
        )

    return bm25_index


def dense_search(question: str, k: int = 5):
    vectors, chunks, model = ensure_collection_loaded()

    response = client.embeddings.create(
        model=model,
        input=[question],
    )

    query = np.asarray(
        response.data[0].embedding,
        dtype=np.float32,
    )

    if query.shape[0] != vectors.shape[1]:
        raise ValueError(
            "Embedding dimension mismatch."
        )

    norm = np.linalg.norm(query)

    if norm == 0:
        raise ValueError(
            "Zero query embedding."
        )

    query /= norm

    scores = vectors @ query

    top_indices = np.argsort(
        scores
    )[::-1][:k]

    return [
        {
            **chunks[int(i)],
            "score": float(scores[i]),
        }
        for i in top_indices
    ]


def bm25_search(question: str, k: int = 5):
    _, chunks, _ = ensure_collection_loaded()

    bm25 = ensure_bm25_loaded()

    query_tokens = tokenize_for_bm25(
        question
    )

    scores = bm25.get_scores(
        query_tokens
    )

    top_indices = np.argsort(
        scores
    )[::-1][:k]

    return [
        {
            **chunks[int(i)],
            "score": float(scores[i]),
        }
        for i in top_indices
    ]


def reciprocal_rank_fusion(
    dense_results: list[dict],
    bm25_results: list[dict],
    k: int = 5,
    rrf_k: int = 60,
    dense_weight: float = 1.0,
    bm25_weight: float = 1.0,
):
    fused = {}

    for rank, result in enumerate(
        dense_results,
        start=1,
    ):
        chunk_id = result["id"]

        if chunk_id not in fused:
            fused[chunk_id] = {
                key: value
                for key, value in result.items()
                if key != "score"
            }

            fused[chunk_id].update({
                "dense_score": None,
                "bm25_score": None,
                "dense_rank": None,
                "bm25_rank": None,
                "rrf_score": 0.0,
            })

        fused[chunk_id]["dense_score"] = (
            result["score"]
        )

        fused[chunk_id]["dense_rank"] = rank

        fused[chunk_id]["rrf_score"] += (
            dense_weight / (rrf_k + rank)
        )

    for rank, result in enumerate(
        bm25_results,
        start=1,
    ):
        chunk_id = result["id"]

        if chunk_id not in fused:
            fused[chunk_id] = {
                key: value
                for key, value in result.items()
                if key != "score"
            }

            fused[chunk_id].update({
                "dense_score": None,
                "bm25_score": None,
                "dense_rank": None,
                "bm25_rank": None,
                "rrf_score": 0.0,
            })

        fused[chunk_id]["bm25_score"] = (
            result["score"]
        )

        fused[chunk_id]["bm25_rank"] = rank

        fused[chunk_id]["rrf_score"] += (
            bm25_weight / (rrf_k + rank)
        )

    ranked = sorted(
        fused.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )

    return ranked[:k]


def hybrid_search(
    question: str,
    k: int = 5,
    dense_k: int | None = None,
    bm25_k: int | None = None,
    rrf_k: int = 60,
    dense_weight: float = 1.0,
    bm25_weight: float = 1.0,
):
    if dense_k is None:
        dense_k = k

    if bm25_k is None:
        bm25_k = k

    dense_results = dense_search(
        question,
        k=dense_k,
    )

    bm25_results = bm25_search(
        question,
        k=bm25_k,
    )

    return reciprocal_rank_fusion(
        dense_results=dense_results,
        bm25_results=bm25_results,
        k=k,
        rrf_k=rrf_k,
        dense_weight=dense_weight,
        bm25_weight=bm25_weight,
    )


# Preserve the existing public function for now.
# This means the rest of the project still uses
# dense-only retrieval until we explicitly switch it.
def search(question: str, k: int = 5):
    return dense_search(
        question,
        k=k,
    )


if __name__ == "__main__":
    while True:
        question = input(
            "\nQuestion: "
        ).strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        print("\n--- DENSE ---")

        for result in dense_search(
            question
        ):
            print(
                f"\n{result['paper']} "
                f"| Page {result['page']} "
                f"| Score {result['score']:.4f}"
            )

            print(
                result["text"][:500]
            )

        print("\n--- BM25 ---")

        for result in bm25_search(
            question
        ):
            print(
                f"\n{result['paper']} "
                f"| Page {result['page']} "
                f"| Score {result['score']:.4f}"
            )

            print(
                result["text"][:500]
            )

        print("\n--- HYBRID / RRF ---")

        for result in hybrid_search(
            question
        ):
            print(
                f"\n{result['paper']} "
                f"| Page {result['page']} "
                f"| RRF {result['rrf_score']:.6f} "
                f"| Dense rank "
                f"{result['dense_rank']} "
                f"| BM25 rank "
                f"{result['bm25_rank']}"
            )

            print(
                result["text"][:500]
            )