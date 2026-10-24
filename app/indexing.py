from pathlib import Path
import argparse
import hashlib
import json
import re

import fitz
import numpy as np
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI


BASE = Path(__file__).resolve().parent
REPO_ROOT = BASE.parent
DEFAULT_SOURCE = BASE / "extracted" / "paper.json"
DEFAULT_PDF_DIR = REPO_ROOT / "papers"
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def paper_id_from_filename(path: Path) -> str:
    paper_id = re.sub(r"[^A-Za-z0-9_-]+", "-", path.stem)
    paper_id = paper_id.strip("-_").lower()

    if not paper_id:
        raise ValueError(f"Could not create a paper ID from: {path.name}")

    return paper_id


def extract_pdf_pages(pdf_path: Path) -> list[dict]:
    pages = []

    with fitz.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()

            if text:
                pages.append({
                    "page": page_number,
                    "text": text,
                })

    if not pages:
        raise ValueError(
            f"No extractable text found in PDF: {pdf_path}. "
            "Scanned PDFs require OCR, which is not handled by this indexer."
        )

    return pages


def load_json_pages(source: Path) -> list[dict]:
    if not source.is_file():
        raise FileNotFoundError(f"Source document not found: {source}")

    with source.open(encoding="utf-8") as f:
        pages = json.load(f)

    if not isinstance(pages, list):
        raise ValueError("JSON source must contain a list of pages.")

    return pages


def build_index_from_pages(
    pages: list[dict],
    index_root: Path,
    paper_id: str,
    *,
    source_name: str | None = None,
    source_sha256: str | None = None,
):
    load_dotenv()
    client = OpenAI()

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
            for item in sorted(response.data, key=lambda item: item.index)
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

    if source_name is not None:
        metadata["source_name"] = source_name

    if source_sha256 is not None:
        metadata["source_sha256"] = source_sha256

    with (paper_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("Embedding matrix:", vectors.shape)
    print("Index saved to:", paper_dir)


def build_index(source: Path, index_root: Path, paper_id: str):
    """Preserve the existing JSON indexing interface."""
    pages = load_json_pages(source)

    build_index_from_pages(
        pages=pages,
        index_root=index_root,
        paper_id=paper_id,
        source_name=source.name,
        source_sha256=file_sha256(source),
    )


def index_pdf(pdf_path: Path, index_root: Path) -> str:
    paper_id = paper_id_from_filename(pdf_path)
    paper_dir = index_root / paper_id
    metadata_path = paper_dir / "metadata.json"
    embeddings_path = paper_dir / "embeddings.npy"
    source_hash = file_sha256(pdf_path)

    if metadata_path.is_file() and embeddings_path.is_file():
        with metadata_path.open(encoding="utf-8") as f:
            metadata = json.load(f)

        unchanged = (
            metadata.get("source_sha256") == source_hash
            and metadata.get("embedding_model") == EMBEDDING_MODEL
            and metadata.get("chunk_size") == CHUNK_SIZE
            and metadata.get("overlap") == CHUNK_OVERLAP
        )

        if unchanged:
            print(f"SKIP {pdf_path.name} (unchanged)")
            return "skipped"

        print(f"REBUILD {pdf_path.name} (source or indexing settings changed)")
    else:
        print(f"INDEX {pdf_path.name}")

    pages = extract_pdf_pages(pdf_path)

    build_index_from_pages(
        pages=pages,
        index_root=index_root,
        paper_id=paper_id,
        source_name=pdf_path.name,
        source_sha256=source_hash,
    )

    return "indexed"


def index_pdf_directory(pdf_dir: Path, index_root: Path):
    if not pdf_dir.is_dir():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    pdf_files = sorted(
        path
        for path in pdf_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdf"
    )

    if not pdf_files:
        raise ValueError(f"No PDF files found in: {pdf_dir}")

    indexed = 0
    skipped = 0

    for pdf_path in pdf_files:
        status = index_pdf(pdf_path, index_root)

        if status == "skipped":
            skipped += 1
        else:
            indexed += 1

    print(f"\nDone. Indexed/rebuilt: {indexed}; skipped: {skipped}.")


def valid_paper_id(value: str) -> bool:
    return bool(value) and all(
        c.isalnum() or c in "-_"
        for c in value
    )


def main():
    parser = argparse.ArgumentParser(
        description="Build embeddings for scientific documents."
    )

    mode = parser.add_mutually_exclusive_group()

    mode.add_argument(
        "--source",
        type=Path,
        help="Index one pre-extracted JSON document.",
    )

    mode.add_argument(
        "--pdf-dir",
        type=Path,
        help="Incrementally index all PDFs in a directory.",
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
        help="Paper ID for --source JSON mode.",
    )

    args = parser.parse_args()

    if args.pdf_dir is not None:
        index_pdf_directory(
            pdf_dir=args.pdf_dir,
            index_root=args.index,
        )
        return

    source = args.source or DEFAULT_SOURCE

    if not valid_paper_id(args.paper_id):
        parser.error(
            "--paper-id must contain only letters, numbers, "
            "hyphens, or underscores."
        )

    build_index(
        source=source,
        index_root=args.index,
        paper_id=args.paper_id,
    )


if __name__ == "__main__":
    main()