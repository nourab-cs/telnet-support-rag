from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


class ChromaStore:
    """Gestionnaire de la base vectorielle Chroma et de son manifest."""

    MANIFEST_FILENAME = "rag_manifest.json"

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "telnet_support",
    ) -> None:
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.vectorstore: Optional[Chroma] = None

    @property
    def manifest_path(self) -> Path:
        return self.persist_directory / self.MANIFEST_FILENAME

    def create_or_replace(
        self,
        chunks: List[Document],
        embeddings: Any,
        documents: Optional[List[Document]] = None,
        embedding_model: str = "",
        chunking_version: str = "",
    ) -> Chroma:
        """Recrée l'index à partir des chunks et persiste ses métadonnées."""
        if not chunks:
            raise ValueError("Impossible de créer Chroma : aucun chunk.")
        if embeddings is None:
            raise ValueError("L'objet embeddings ne peut pas être None.")

        # Le répertoire doit être dédié à Chroma.
        if self.persist_directory.exists():
            shutil.rmtree(self.persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        ids: List[str] = []
        used_ids: set[str] = set()
        for index, chunk in enumerate(chunks):
            metadata = chunk.metadata or {}
            chunk_id = str(metadata.get("chunk_id") or "").strip()
            if not chunk_id:
                document_id = str(metadata.get("document_id") or metadata.get("source") or "document")
                chunk_id = f"{document_id}:chunk:{index}"
                metadata["chunk_id"] = chunk_id
                chunk.metadata = metadata
            # Chroma exige des IDs uniques.
            if chunk_id in used_ids:
                chunk_id = f"{chunk_id}:dup:{index}"
                chunk.metadata["chunk_id"] = chunk_id
            used_ids.add(chunk_id)
            ids.append(chunk_id)

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            ids=ids,
            collection_name=self.collection_name,
            persist_directory=str(self.persist_directory),
        )

        manifest = {
            "collection_name": self.collection_name,
            "embedding_model": embedding_model,
            "chunking_version": chunking_version,
            "document_count": len(documents or []),
            "chunk_count": len(chunks),
        }
        self.manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return self.vectorstore

    def load(self, embeddings) -> Chroma:
        if not self.persist_directory.exists():
            raise FileNotFoundError(f"Base Chroma introuvable : {self.persist_directory}")

        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=embeddings,
            persist_directory=str(self.persist_directory),
        )

        try:
            count = self.vectorstore._collection.count()
        except Exception:
            count = None

        if count == 0:
            raise ValueError("La collection Chroma existe mais ne contient aucun chunk.")

        return self.vectorstore

    def get_documents(self) -> List[Document]:
        """Recharge les chunks depuis Chroma, notamment pour BM25."""
        vectorstore = self.get()
        data = vectorstore.get(include=["documents", "metadatas"])
        texts = data.get("documents") or []
        metadatas = data.get("metadatas") or []
        return [
            Document(
                page_content=text or "",
                metadata=dict(metadatas[i] or {}) if i < len(metadatas) else {},
            )
            for i, text in enumerate(texts)
            if text is not None
        ]

    def get_manifest(self) -> Dict[str, Any]:
        if not self.manifest_path.exists():
            return {}
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {}

    def is_compatible(self, embedding_model: str, chunking_version: str) -> bool:
        manifest = self.get_manifest()
        if not manifest:
            return False
        return (
            manifest.get("embedding_model") == embedding_model
            and manifest.get("chunking_version") == chunking_version
            and manifest.get("collection_name") == self.collection_name
        )

    def get(self) -> Chroma:
        if self.vectorstore is None:
            raise RuntimeError("Chroma n'est pas initialisé.")
        return self.vectorstore
