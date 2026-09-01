"""
Chunking sémantique utilisant LangChain Experimental.

Regroupe les phrases selon leur similarité sémantique
plutôt que leur position dans le document.
"""

from typing import List
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings


class SemanticChunker:
    """
    Chunking sémantique utilisant LangChain Experimental.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_amount: float = 70,
    ):
        """
        Initialise le chunker sémantique.

        Args:
            model_name: Modèle d'embeddings (BGE-M3 par défaut)
            breakpoint_threshold_type: Type de seuil ("percentile", "standard_deviation", "gradient")
            breakpoint_threshold_amount: Valeur du seuil (70 pour percentile)
        """
        # Créer les embeddings avec BGE-M3
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Initialiser le SemanticChunker de LangChain
        self.splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount,
        )

    def chunk_document(self, document: Document) -> List[Document]:
        """
        Applique le chunking sémantique à un document.

        Args:
            document: Document LangChain

        Returns:
            Liste de documents chunkés
        """
        text = document.page_content

        # Appliquer le chunking sémantique
        chunk_texts = self.splitter.split_text(text)

        # Créer les documents
        chunks = []
        for i, chunk_text in enumerate(chunk_texts):
            chunks.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        **document.metadata,
                        "chunk_index": i,
                        "chunking_method": "semantic",
                        "chunk_size": len(chunk_text)
                    }
                )
            )

        return chunks

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Applique le chunking sémantique à une liste de documents.

        Args:
            documents: Liste de documents LangChain

        Returns:
            Liste de tous les documents chunkés
        """
        all_chunks = []

        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)

        print(f"Chunking sémantique: {len(all_chunks)} chunks générés")

        return all_chunks
