from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document


class ChromaStore:
    """
    Gestion de la base vectorielle ChromaDB.

    Responsabilités :
    - vérifier l'existence de l'index
    - calculer la signature du corpus
    - vérifier la compatibilité
    - créer/reconstruire l'index
    - charger un index existant
    - gérer le manifest
    - supprimer l'index
    """

    MANIFEST_VERSION = 1
    MANIFEST_FILENAME = "index_manifest.json"

    def __init__(
        self,
        persist_directory: str,
        collection_name: str = "telnet_support",
    ):
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name

    # ============================================================
    # MANIFEST
    # ============================================================

    @property
    def manifest_path(self) -> Path:
        return self.persist_directory / self.MANIFEST_FILENAME

    def exists(self) -> bool:
        """
        Vérifie si un index Chroma complet existe.
        """

        return (
            self.persist_directory.exists()
            and self.manifest_path.exists()
        )

    def get_manifest(self) -> dict:
        """
        Retourne le manifest actuel.
        """

        if not self.manifest_path.exists():
            return {}

        try:
            return self._load_manifest()
        except Exception:
            return {}

    # ============================================================
    # SIGNATURE DU CORPUS
    # ============================================================

    def compute_corpus_signature(
        self,
        documents: List[Document],
    ) -> str:
        """
        Calcule une signature stable du corpus.

        La signature dépend :
        - du chemin du document
        - du hash du document
        """

        entries = []
        seen = set()

        for document in documents:

            relative_path = document.metadata.get(
                "relative_path",
                document.metadata.get(
                    "source",
                    "",
                ),
            )

            file_hash = document.metadata.get(
                "file_hash",
                "",
            )

            key = (
                relative_path,
                file_hash,
            )

            if key in seen:
                continue

            seen.add(key)

            entries.append(
                {
                    "path": relative_path,
                    "file_hash": file_hash,
                }
            )

        entries.sort(
            key=lambda item: item["path"]
        )

        payload = json.dumps(
            entries,
            ensure_ascii=False,
            sort_keys=True,
        )

        return hashlib.sha256(
            payload.encode("utf-8")
        ).hexdigest()

    # ============================================================
    # COMPATIBILITÉ
    # ============================================================

    def is_compatible(
        self,
        documents: List[Document],
        embedding_model: str,
        chunking_version: str,
    ) -> bool:
        """
        Vérifie si l'index actuel correspond
        au corpus et aux paramètres d'indexation.
        """

        if not self.exists():
            return False

        try:

            manifest = self._load_manifest()

            current_signature = (
                self.compute_corpus_signature(
                    documents
                )
            )

            return (
                manifest.get("version")
                == self.MANIFEST_VERSION
                and manifest.get("collection_name")
                == self.collection_name
                and manifest.get("embedding_model")
                == embedding_model
                and manifest.get("chunking_version")
                == chunking_version
                and manifest.get("corpus_signature")
                == current_signature
            )

        except Exception:
            return False

    # ============================================================
    # CHARGEMENT
    # ============================================================

    def load(
        self,
        embeddings,
    ) -> Chroma:
        """
        Charge une base Chroma existante.
        """

        if not self.persist_directory.exists():
            raise FileNotFoundError(
                "La base Chroma n'existe pas."
            )

        if not self.manifest_path.exists():
            raise FileNotFoundError(
                "Manifest Chroma introuvable."
            )

        return Chroma(
            collection_name=self.collection_name,
            embedding_function=embeddings,
            persist_directory=str(
                self.persist_directory
            ),
        )

    # ============================================================
    # CREATION
    # ============================================================

    def create_or_replace(
        self,
        chunks: List[Document],
        embeddings,
        documents: Optional[List[Document]] = None,
        embedding_model: str = "",
        chunking_version: str = "",
    ) -> Chroma:
        """
        Supprime l'ancien index et crée un nouvel index.
        """

        if not chunks:
            raise ValueError(
                "Impossible de créer Chroma sans chunks."
            )

        # Supprimer ancien index
        if self.persist_directory.exists():
            shutil.rmtree(
                self.persist_directory,
                ignore_errors=True,
            )

        self.persist_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        # IDs déterministes
        ids = self._build_chunk_ids(chunks)

        print(
            f"Création de Chroma avec "
            f"{len(chunks)} chunks..."
        )

        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            ids=ids,
            collection_name=self.collection_name,
            persist_directory=str(
                self.persist_directory
            ),
        )

        # Manifest
        manifest = {
            "version": self.MANIFEST_VERSION,
            "collection_name": self.collection_name,
            "embedding_model": embedding_model,
            "chunking_version": chunking_version,
            "corpus_signature": (
                self.compute_corpus_signature(
                    documents
                )
                if documents
                else None
            ),
            "document_count": (
                len(documents)
                if documents
                else None
            ),
            "chunk_count": len(chunks),
            "created_at": datetime.now().isoformat(),
        }

        self._save_manifest(manifest)

        print(
            f"[OK] Base vectorielle créée : "
            f"{len(chunks)} chunks"
        )

        return vectordb

    # ============================================================
    # IDS
    # ============================================================

    @staticmethod
    def _build_chunk_ids(
        chunks: List[Document],
    ) -> List[str]:
        """
        Génère des IDs déterministes.
        """

        ids = []

        for index, chunk in enumerate(chunks):

            chunk_id = chunk.metadata.get(
                "chunk_id"
            )

            if not chunk_id:

                source = chunk.metadata.get(
                    "source",
                    "unknown",
                )

                content_hash = hashlib.sha256(
                    chunk.page_content.encode(
                        "utf-8"
                    )
                ).hexdigest()[:16]

                chunk_id = (
                    f"{source}:{index}:{content_hash}"
                )

            ids.append(chunk_id)

        return ids

    # ============================================================
    # MANIFEST INTERNE
    # ============================================================

    def _save_manifest(
        self,
        manifest: dict,
    ) -> None:

        self.persist_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.manifest_path.write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _load_manifest(self) -> dict:

        return json.loads(
            self.manifest_path.read_text(
                encoding="utf-8"
            )
        )

    # ============================================================
    # SUPPRESSION
    # ============================================================

    def delete(self) -> None:
        """
        Supprime complètement l'index.
        """

        if self.persist_directory.exists():

            shutil.rmtree(
                self.persist_directory,
                ignore_errors=True,
            )

        print("[OK] Index Chroma supprimé.")
