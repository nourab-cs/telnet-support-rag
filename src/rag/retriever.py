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
    - mmr
    - similarity_score_threshold
    - recherche avec scores
    - Relevance Gate
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
        score_threshold: float = 0.40,
        fetch_k: int = 20,
        lambda_mult: float = 0.6,
    ):
        if vectordb is None:
            raise ValueError(
                "vectordb ne peut pas être None."
            )

        if k <= 0:
            raise ValueError(
                "k doit être supérieur à 0."
            )

        if search_type not in self.VALID_SEARCH_TYPES:
            raise ValueError(
                f"search_type invalide : {search_type}. "
                f"Valeurs autorisées : {self.VALID_SEARCH_TYPES}"
            )

        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError(
                "score_threshold doit être compris entre 0 et 1."
            )

        if fetch_k <= 0:
            raise ValueError(
                "fetch_k doit être supérieur à 0."
            )

        if not 0.0 <= lambda_mult <= 1.0:
            raise ValueError(
                "lambda_mult doit être compris entre 0 et 1."
            )

        self.vectordb = vectordb
        self.k = k
        self.search_type = search_type
        self.score_threshold = score_threshold
        self.fetch_k = max(fetch_k, k)
        self.lambda_mult = lambda_mult

        self.retriever = self._build_retriever()

        logger.info(
            "Retriever initialisé | "
            "type=%s | k=%d | fetch_k=%d | "
            "threshold=%.2f | lambda=%.2f",
            self.search_type,
            self.k,
            self.fetch_k,
            self.score_threshold,
            self.lambda_mult,
        )

    # ============================================================
    # CONSTRUCTION
    # ============================================================

    def _build_retriever(self):
        """
        Construit le retriever LangChain.
        """

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
        """
        Effectue une recherche documentaire.

        Respecte le search_type configuré.
        """

        if not query or not query.strip():
            logger.warning(
                "Recherche avec une requête vide."
            )
            return []

        query = query.strip()

        try:
            documents = self.retriever.invoke(query)

        except Exception:
            logger.exception(
                "Erreur pendant la recherche."
            )
            raise

        logger.info(
            "Retrieval [%s] : %d documents.",
            self.search_type,
            len(documents),
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
        """
        Recherche par similarité avec scores.

        Cette méthode est utilisée pour le Relevance Gate.

        Retour :
            [
                (document, score),
                ...
            ]
        """

        if not query or not query.strip():
            logger.warning(
                "Recherche avec une requête vide."
            )
            return []

        query = query.strip()

        effective_k = k or self.k

        if effective_k <= 0:
            raise ValueError(
                "k doit être supérieur à 0."
            )

        try:
            results = (
                self.vectordb
                .similarity_search_with_relevance_scores(
                    query,
                    k=effective_k,
                )
            )

        except Exception:
            logger.exception(
                "Erreur pendant la recherche avec scores."
            )
            raise

        logger.info(
            "Similarity avec scores : %d documents.",
            len(results),
        )

        for document, score in results:

            source = document.metadata.get(
                "source",
                document.metadata.get(
                    "file_name",
                    "unknown",
                ),
            )

            logger.debug(
                "Score=%.4f | source=%s",
                score,
                source,
            )

        return results

    # ============================================================
    # RELEVANCE GATE
    # ============================================================

    def retrieve_relevant(
        self,
        query: str,
        k: int | None = None,
        threshold: float | None = None,
    ) -> List[Tuple[Document, float]]:
        """
        Recherche puis applique le Relevance Gate.

        Seuls les documents ayant :

            score >= threshold

        sont conservés.
        """

        effective_threshold = (
            self.score_threshold
            if threshold is None
            else threshold
        )

        if not 0.0 <= effective_threshold <= 1.0:
            raise ValueError(
                "threshold doit être compris entre 0 et 1."
            )

        results = self.retrieve_with_scores(
            query=query,
            k=k or self.k,
        )

        relevant = [
            (document, score)
            for document, score in results
            if score >= effective_threshold
        ]

        logger.info(
            "Relevance Gate : %d/%d documents conservés "
            "avec threshold=%.2f.",
            len(relevant),
            len(results),
            effective_threshold,
        )

        return relevant

    # ============================================================
    # STATISTIQUES DES SCORES
    # ============================================================

    @staticmethod
    def get_score_statistics(
        results: List[Tuple[Document, float]],
    ) -> dict:
        """
        Calcule les statistiques des scores.

        Utile pour calibrer le threshold.
        """

        if not results:
            return {
                "count": 0,
                "min": None,
                "max": None,
                "mean": None,
            }

        scores = [
            float(score)
            for _, score in results
        ]

        return {
            "count": len(scores),
            "min": min(scores),
            "max": max(scores),
            "mean": sum(scores) / len(scores),
        }