"""Quick manual test: run the full RAG pipeline on sample data files."""
from src.data.loader import load_documents, search_by_title
from src.data.chunker import TokenChunker
from src.rag.embedder import SentenceTransformerEmbedder
from src.rag.vectorstore import FAISSVectorStore
from src.rag.retriever import Retriever


def test_with_dir(label: str, data_dir: str):
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")

    # Load
    docs = load_documents(data_dir)
    print(f"\nLoaded {len(docs)} documents:")
    for d in docs:
        print(f"  [{d.doc_id}] {d.title}  (year={d.year}, authors={d.authors})")

    # Title search
    print("\nTitle search for 'Neural':")
    for d in search_by_title("Neural", docs):
        print(f"  -> {d.title}")

    # Chunk
    chunker = TokenChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk_documents(docs)
    print(f"\nChunked into {len(chunks)} chunks")

    # Embed + Index
    embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")
    store = FAISSVectorStore()
    embeddings = embedder.embed([c.text for c in chunks])
    store.add(
        [c.chunk_id for c in chunks],
        [c.text for c in chunks],
        embeddings,
        [{"doc_id": c.doc_id, "title": c.metadata.get("title", "")} for c in chunks],
    )
    print(f"Indexed {store.count()} vectors in FAISS")

    # Retrieve
    retriever = Retriever(embedder=embedder, vectorstore=store, top_k=3)
    questions = [
        "What is reinforcement learning?",
        "How does the Transformer architecture work?",
        "What are graph neural networks?",
    ]
    for q in questions:
        results = retriever.retrieve(q)
        print(f"\nQ: {q}")
        for r in results:
            doc_id = r.chunk_id.split("__")[0]
            print(f"  score={r.score:.4f}  doc={doc_id}  chunk={r.chunk_id}")

    print(f"\n>>> {label} PASSED <<<")


if __name__ == "__main__":
    test_with_dir("JSON sample files", "data/sample/json")
    test_with_dir("TXT sample files", "data/sample/txt")
    print("\n" + "=" * 60)
    print("  ALL MANUAL TESTS PASSED")
    print("=" * 60)
