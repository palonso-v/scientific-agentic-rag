
from pathlib import Path
import json
import argparse

import numpy as np
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI


BASE = Path(__file__).resolve().parent
DEFAULT_SOURCE = BASE / "extracted" / "paper.json"
DEFAULT_INDEX = BASE / "index_v2"

EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 60

encoding = tiktoken.get_encoding("cl100k_base")


def split_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Invalid chunk size or overlap.")

    tokens = encoding.encode(text)
    step = size - overlap

    return [
        encoding.decode(tokens[i:i + size])
        for i in range(0, len(tokens), step)
        if tokens[i:i + size]
    ]


def build_index(source: Path, index_root: Path, paper_id: str):
    load_dotenv()
    client = OpenAI()

    if not source.is_file():
        raise FileNotFoundError(
            f"Source document not found: {source}"
        )

    with source.open(encoding="utf-8") as f:
        pages = json.load(f)

    chunks = []

    for page in pages:
        page_number = page["page"]
        page_text = page["text"]

        for j, text in enumerate(split_text(page_text), start=1):
            chunks.append({
                "id": f"{paper_id}_p{page_number}_c{j}",
                "paper": paper_id,
                "page": page_number,
                "text": text,
            })

    if not chunks:
        raise ValueError("No text found to index.")

    print(f"Created {len(chunks)} chunks.")

    vectors = []
    batch_size = 32

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]

        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[item["text"] for item in batch],
        )

        batch_vectors = [
            item.embedding
            for item in sorted(
                response.data,
                key=lambda item: item.index,
            )
        ]

        vectors.extend(batch_vectors)

    vectors = np.asarray(vectors, dtype=np.float32)

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)

    if np.any(norms == 0):
        raise ValueError("Encountered a zero embedding.")

    vectors = vectors / norms

    if len(chunks) != len(vectors):
        raise ValueError("Chunk/embedding count mismatch.")

    paper_dir = index_root / paper_id
    paper_dir.mkdir(parents=True, exist_ok=True)

    np.save(paper_dir / "embeddings.npy", vectors)

    metadata = {
        "paper_id": paper_id,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": vectors.shape[1],
        "chunk_size": CHUNK_SIZE,
        "overlap": CHUNK_OVERLAP,
        "chunks": chunks,
    }

    with (paper_dir / "metadata.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("Embedding matrix:", vectors.shape)
    print("Index saved to:", paper_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Build embeddings for a scientific document."
    )

    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
    )

    parser.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX,
    )

    parser.add_argument(
        "--paper-id",
        type=str,
        default="paper",
    )

    args = parser.parse_args()

    if not args.paper_id or not all(
        c.isalnum() or c in "-_"
        for c in args.paper_id
    ):
        parser.error(
            "--paper-id must contain only letters, numbers, "
            "hyphens, or underscores."
        )

    build_index(
        source=args.source,
        index_root=args.index,
        paper_id=args.paper_id,
    )


if __name__ == "__main__":
    main()
