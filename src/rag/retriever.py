from __future__ import annotations

import logging
from typing import List, Tuple

from langchain_core.documents import Document


logger = logging.getLogger(__name__)


class Retriever:
    """
    Retriever pour Chroma.

    Supporte :
    - similarity
    - MMR
    - similarity avec threshold
    - récupération avec scores
    - filtrage par pertinence
    """

    VALID_SEARCH_TYPES = {
        "similarity",
        "mmr",
        "similarity_score_threshold",
    }

    def __init__(
        self,
        vectordb,
        k: int = 8,
        search_type: str = "mmr",
        score_threshold: float = 0.65,
        fetch_k: int = 20,
        lambda_mult: float = 0.6,
    ):

        if k <= 0:
            raise ValueError(
                "k doit être > 0."
            )

        if search_type not in self.VALID_SEARCH_TYPES:
            raise ValueError(
                f"search_type invalide : {search_type}"
            )

        if not 0 <= lambda_mult <= 1:
            raise ValueError(
                "lambda_mult doit être entre 0 et 1."
            )

        self.vectordb = vectordb
        self.k = k
        self.search_type = search_type
        self.score_threshold = score_threshold
        self.fetch_k = max(fetch_k, k)
        self.lambda_mult = lambda_mult

        self.retriever = self._build_retriever()

    # ============================================================
    # CONSTRUCTION
    # ============================================================

    def _build_retriever(self):

        if self.search_type == "mmr":

            return self.vectordb.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": self.k,
                    "fetch_k": self.fetch_k,
                    "lambda_mult": self.lambda_mult,
                },
            )

        if self.search_type == "similarity_score_threshold":

            return self.vectordb.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": self.k,
                    "score_threshold": self.score_threshold,
                },
            )

        return self.vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": self.k,
            },
        )

    # ============================================================
    # RETRIEVAL
    # ============================================================

    def retrieve(
        self,
        query: str,
    ) -> List[Document]:

        if not query or not query.strip():
            return []

        query = query.strip()

        documents = self.retriever.invoke(
            query
        )

        logger.info(
            f"Retrieval : {len(documents)} documents."
        )

        return documents

    # ============================================================
    # RETRIEVAL AVEC SCORES
    # ============================================================

    def retrieve_with_scores(
        self,
        query: str,
        k: int | None = None,
    ) -> List[Tuple[Document, float]]:

        if not query or not query.strip():
            return []

        return (
            self.vectordb
            .similarity_search_with_relevance_scores(
                query.strip(),
                k=k or self.k,
            )
        )

    # ============================================================
    # FILTRAGE
    # ============================================================

    def retrieve_relevant(
        self,
        query: str,
        k: int | None = None,
        threshold: float | None = None,
    ) -> List[Document]:

        threshold = (
            threshold
            if threshold is not None
            else self.score_threshold
        )

        results = self.retrieve_with_scores(
            query=query,
            k=k or self.k,
        )

        relevant = [
            document
            for document, score in results
            if score >= threshold
        ]

        logger.info(
            f"{len(relevant)}/{len(results)} "
            f"documents dépassent le seuil {threshold}."
        )

        return relevant
