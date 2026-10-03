"""Embed chunks with sentence-transformers and store/query them in Chroma DB.

Usage (from the repo root):
    python src/vector_store.py build                 # embed data/chunks.json -> data/chroma_db
    python src/vector_store.py query "What is RAG?"  # top-5 chunks for a question
"""
import argparse
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = ROOT / "data" / "chunks.json"
DB_PATH = ROOT / "data" / "chroma_db"
COLLECTION_NAME = "papers"

# Chosen in notebooks/embedding_and_storage.ipynb
MODEL_NAME = "BAAI/bge-small-en-v1.5"
# BGE models retrieve better when queries (not passages) carry this instruction
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def load_chunks(path=CHUNKS_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def chunk_ids(chunks):
    """Stable IDs like '2005.11401_0007', numbered per paper in file order."""
    ids, counters = [], {}
    for c in chunks:
        pid = c["metadata"]["paper_id"]
        n = counters.get(pid, 0)
        counters[pid] = n + 1
        ids.append(f"{pid}_{n:04d}")
    return ids


def get_client(db_path=DB_PATH):
    return chromadb.PersistentClient(path=str(db_path))


def build_collection(chunks_path=CHUNKS_PATH, db_path=DB_PATH, batch_size=64):
    """Embed every chunk and (re)create the Chroma collection from scratch."""
    chunks = load_chunks(chunks_path)
    ids = chunk_ids(chunks)
    texts = [c["page_content"] for c in chunks]
    metadatas = []
    for i, c in zip(ids, chunks):
        meta = dict(c["metadata"])
        meta["chunk_index"] = int(i.rsplit("_", 1)[1])
        meta["embedding_model"] = MODEL_NAME
        metadatas.append(meta)

    embeddings = get_model().encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    client = get_client(db_path)
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "embedding_model": MODEL_NAME},
    )

    # Chroma caps the size of a single add() call, so insert in batches
    step = 500
    for s in range(0, len(ids), step):
        collection.add(
            ids=ids[s:s + step],
            documents=texts[s:s + step],
            embeddings=embeddings[s:s + step].tolist(),
            metadatas=metadatas[s:s + step],
        )
    return collection


def get_collection(db_path=DB_PATH):
    return get_client(db_path).get_collection(COLLECTION_NAME)


def query(question, k=5, where=None, collection=None):
    """Return the top-k chunks for a question as a list of dicts.

    `where` is an optional Chroma metadata filter, e.g. {"paper_id": "2005.11401"}.
    `distance` is cosine distance (lower = more similar).
    """
    collection = collection or get_collection()
    q_emb = get_model().encode([QUERY_PREFIX + question], normalize_embeddings=True)
    res = collection.query(
        query_embeddings=q_emb.tolist(),
        n_results=k,
        where=where,
    )
    return [
        {"id": i, "distance": d, "text": doc, "metadata": m}
        for i, d, doc, m in zip(
            res["ids"][0], res["distances"][0], res["documents"][0], res["metadatas"][0]
        )
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", help="embed chunks.json and store in Chroma")
    q = sub.add_parser("query", help="retrieve chunks for a question")
    q.add_argument("question")
    q.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    if args.cmd == "build":
        col = build_collection()
        print(f"Stored {col.count()} chunks in '{COLLECTION_NAME}' at {DB_PATH}")
    else:
        for r in query(args.question, k=args.k):
            m = r["metadata"]
            print(f"[{r['distance']:.3f}] {r['id']}  {m['title']} (p.{m['page_number']}-{m['end_page']})")
            print("    " + r["text"][:200].replace("\n", " ") + "...\n")


if __name__ == "__main__":
    main()
