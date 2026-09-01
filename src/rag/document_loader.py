from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader


class DocumentLoader:

    def __init__(self, data_dir: str):

        self.data_dir = Path(data_dir)

    def load(self) -> List[Document]:

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Le dossier '{self.data_dir}' n'existe pas."
            )

        if not self.data_dir.is_dir():
            raise NotADirectoryError(
                f"'{self.data_dir}' n'est pas un dossier."
            )

        documents = []

        print("=" * 60)
        print("Chargement des documents")
        print("=" * 60)

        # ---------- MD ----------
        md_loader = DirectoryLoader(
            str(self.data_dir),
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True,
        )
        documents.extend(md_loader.load())

        # ---------- TXT ----------
        txt_loader = DirectoryLoader(
            str(self.data_dir),
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True,
        )
        documents.extend(txt_loader.load())

        # ---------- PDF ----------
        pdf_loader = DirectoryLoader(
            str(self.data_dir),
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
        )
        documents.extend(pdf_loader.load())

        # ---------- DOC/DOCX ----------
        doc_loader = DirectoryLoader(
            str(self.data_dir),
            glob="**/*.doc*",
            loader_cls=UnstructuredWordDocumentLoader,
            show_progress=True,
        )
        documents.extend(doc_loader.load())

        print(f"\nDocuments trouvés : {len(documents)}")

        # ---------- Nettoyage ----------
        cleaned_documents = []

        for doc in documents:
            # Récupérer le texte
            text = doc.page_content.strip()

            # Ignorer les documents trop courts
            if len(text) < 30:
                continue

            # Nettoyage léger sans détruire la structure
            text = "\n".join(
                line.strip()
                for line in text.splitlines()
                if line.strip()
            )

            # Mettre à jour le contenu
            doc.page_content = text

            # Récupérer la source
            source = doc.metadata.get("source", "")

            # Métadonnées
            doc.metadata = {
                "source": source,
                "filename": Path(source).name,
                "extension": Path(source).suffix.lower(),
            }

            # Ajouter le document nettoyé
            cleaned_documents.append(doc)

        print(f"{len(cleaned_documents)} documents chargés.")

        return cleaned_documents