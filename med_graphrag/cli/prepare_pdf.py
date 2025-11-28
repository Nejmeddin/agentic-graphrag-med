from med_graphrag.config.settings import settings
from med_graphrag.data_pipeline.doc_loader import process_all_documents
from med_graphrag.data_pipeline.chunker import split_documents, save_chunks_jsonl


def main():
    data_dir = "C:\\Users\\User\\Desktop\\agentic-graphrag-med\\data\\source"

    print("📂 Loading documents (PDF/TXT) with LangChain loaders...")
    documents = process_all_documents(data_dir)
    print(f"  → Loaded {len(documents)} base documents")

    print("✂️  Splitting into chunks...")
    chunks = split_documents(
        documents,
        chunk_size=1000,
        chunk_overlap=200,
    )

    print("💾 Saving chunks as JSONL...")
    output_path = save_chunks_jsonl(chunks)
    print(f"✅ Done. Chunks file: {output_path}")


if __name__ == "__main__":
    main()
