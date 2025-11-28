from pathlib import Path
from typing import List

from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_core.documents import Document  # type: ignore


def process_all_documents(data_directory: str) -> List[Document]:
    """
    Process all PDF and TXT files in a directory (recursively), like in your notebook.

    Returns a flat list of LangChain Document objects with metadata:
      - page (for PDFs, when loaded in page mode)
      - source (file path)
    """
    all_documents: List[Document] = []
    data_dir = Path(data_directory)

    # Find all PDF and TXT files recursively
    files = list(data_dir.rglob("*.pdf")) + list(data_dir.rglob("*.txt"))

    print(f"📂 Found {len(files)} files to process in {data_dir}")

    for file_path in files:
        try:
            if file_path.suffix.lower() == ".pdf":
                # Use mode="page" to keep page-level metadata, as recommended :contentReference[oaicite:4]{index=4}
                loader = PyPDFLoader(str(file_path), mode="page")
                docs = loader.load()
                print(f"  ✅ {file_path.name}: {len(docs)} pages loaded")
                all_documents.extend(docs)
            elif file_path.suffix.lower() == ".txt":
                loader = TextLoader(str(file_path), encoding="utf-8")
                docs = loader.load()
                print(f"  ✅ {file_path.name}: {len(docs)} text docs loaded")
                all_documents.extend(docs)
        except Exception as e:
            print(f"  ❌ Error loading {file_path.name}: {e}")

    print(f"📄 Total documents loaded: {len(all_documents)}")
    return all_documents