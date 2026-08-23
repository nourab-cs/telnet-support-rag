from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader


class DocumentLoader:

    def __init__(self, data_dir: str):

        self.data_dir = Path(data_dir)

    def load(self) -> List[Document]:

        documents = []

        print("=" * 60)
        print("Chargement des documents")
        print("=" * 60)

        # Charger tous les fichiers texte (md, txt, etc.)
        loader = DirectoryLoader(
            str(self.data_dir),
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True,
        )
        
        documents = loader.load()

        # ---------- Nettoyage ----------
        cleaned_documents = []

        for doc in documents:

            text = doc.page_content.strip()

            if len(text) < 30:
                continue

            text = " ".join(text.split())

            doc.page_content = text

            source = doc.metadata.get("source", "")

            doc.metadata = {
                "source": source,
                "filename": Path(source).name,
                "extension": Path(source).suffix,
            }

            cleaned_documents.append(doc)

        print(f"{len(cleaned_documents)} documents chargés.")

        return cleaned_documents