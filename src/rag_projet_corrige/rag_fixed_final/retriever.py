
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


logger = logging.getLogger(__name__)


class Retriever:
    """
    Retriever RAG TELNET SmartConnect.

    Méthodes supportées :
    - similarity
    - mmr
    - similarity_score_threshold
    - bm25
    - hybrid

    Pour BM25 et Hybrid, les scores ne sont PAS des scores vectoriels.
    Ils doivent donc être calibrés séparément dans le Pipeline.
    """

    VALID_SEARCH_TYPES = {
        "similarity",
        "mmr",
        "similarity_score_threshold",
        "bm25",
        "hybrid",
    }

    def __init__(
        self,
        vectordb: Any, 
        documents: Optional[List[Document]] = None, #bm25, hybrid
        k: int = 8, 
        search_type: str = "similarity",
        fetch_k: int = 20,    #mmr
        lambda_mult: float = 0.6, #mmr
        hybrid_k: int = 10,   #hybrid
        similarity_score_threshold: Optional[float] = None, #similarity_score_threshold
    ) -> None:

        if vectordb is None:
            raise ValueError("vectordb ne peut pas être None.")

        if search_type not in self.VALID_SEARCH_TYPES:
            raise ValueError(
                f"search_type='{search_type}' invalide. "
                f"Valeurs autorisées : {sorted(self.VALID_SEARCH_TYPES)}"
            )

        if int(k) <= 0:
            raise ValueError("k doit être > 0.")

        if int(fetch_k) <= 0:
            raise ValueError("fetch_k doit être > 0.")

        if not 0.0 <= float(lambda_mult) <= 1.0:
            raise ValueError("lambda_mult doit être compris entre 0 et 1.")

        if int(hybrid_k) <= 0:
            raise ValueError("hybrid_k doit être > 0.")

        if similarity_score_threshold is not None:
            if not 0.0 <= float(similarity_score_threshold) <= 1.0:
                raise ValueError(
                    "similarity_score_threshold doit être compris entre 0 et 1."
                )

        if search_type in ("bm25", "hybrid") and not documents:
            raise ValueError(
                f"La recherche {search_type} nécessite la liste des chunks "
                "dans 'documents'."
            )

        self.vectordb = vectordb
        self.documents = list(documents or [])

        self.k = int(k)
        self.search_type = search_type
        self.fetch_k = int(fetch_k)
        self.lambda_mult = float(lambda_mult)
        self.hybrid_k = int(hybrid_k)

        self.similarity_score_threshold = (
            None
            if similarity_score_threshold is None
            else float(similarity_score_threshold)
        )

        self.bm25 = None
        self.bm25_documents: List[Document] = []

        if self.search_type in ("bm25", "hybrid"):
            self._initialize_bm25()

        self.retriever = self._build_retriever()

    # ============================================================
    # TOKENISATION BM25
    # ============================================================

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Tokenisation adaptée aux documents techniques.

        Exemple :
            https://host/api/devices
        produit notamment :
            https://host/api/devices
            https
            host
            api
            devices

        tout en conservant aussi les tokens techniques complets.
        """

        if not text:
            return []

        text = str(text).lower()

        # Capturer les tokens techniques complets (URLs, chemins, identifiants techniques)
        # Cette regex capture des séquences comme "https://host/api/devices" ou "TLS-8443"
        full_tokens = re.findall(
            r"[a-zàâçéèêëîïôûùüÿñæœ0-9_-]+"
            r"(?:://?[a-zàâçéèêëîïôûùüÿñæœ0-9_/-]+)*"  # Pour https://host/...
            r"(?:[./-][a-zàâçéèêëîïôûùüÿñæœ0-9_/-]+)*",  # Pour /api/devices ou TLS-8443
            text,
        )

        # Capturer les tokens atomiques individuels
        atomic_tokens = re.findall(
            r"[a-zàâçéèêëîïôûùüÿñæœ0-9_]+",
            text,
        )

        tokens: List[str] = []
        seen = set()

        for token in full_tokens + atomic_tokens:
            if token not in seen:
                seen.add(token)
                tokens.append(token)

        return tokens

    # ============================================================
    # INITIALISATION BM25
    # ============================================================

    def _initialize_bm25(self) -> None:

        if BM25Okapi is None:
            raise ImportError(
                "rank-bm25 n'est pas installé.\n"
                "Installe-le avec : pip install rank-bm25"
            )

        self.bm25_documents = list(self.documents)

        tokenized_documents = [
            self._tokenize(doc.page_content or "")
            for doc in self.bm25_documents
        ]

        if not any(tokenized_documents):
            raise ValueError(
                "Aucun token exploitable n'a été trouvé dans les documents BM25."
            )

        self.bm25 = BM25Okapi(tokenized_documents)

        logger.info(
            "Index BM25 initialisé | documents=%d",
            len(self.bm25_documents),
        )

    # ============================================================
    # CONSTRUCTION RETRIEVER LANGCHAIN
    # ============================================================

    def _build_retriever(self):

        if self.search_type in ("bm25", "hybrid"):
            return None

        if self.search_type == "mmr":
            return self.vectordb.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": self.k,
                    "fetch_k": max(self.fetch_k, self.k),
                    "lambda_mult": self.lambda_mult,
                },
            )

        if self.search_type == "similarity":
            return self.vectordb.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": self.k,
                },
            )

        if self.search_type == "similarity_score_threshold":
            # Le threshold est appliqué explicitement dans
            # retrieve_with_scores(). Cela évite les différences
            # de comportement entre versions de LangChain/Chroma.
            return self.vectordb.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": self.k,
                },
            )

        raise RuntimeError(
            f"Type de recherche non supporté : {self.search_type}"
        )

    # ============================================================
    # OUTILS DOCUMENT
    # ============================================================

    @staticmethod
    def _document_key(document: Document) -> Tuple[str, str, str]:
        metadata = document.metadata or {}

        source = str(
            metadata.get("source")
            or metadata.get("relative_path")
            or metadata.get("file_path")
            or metadata.get("filename")
            or ""
        )

        chunk_id = str(
            metadata.get("chunk_id")
            or ""
        )

        content = str(
            document.page_content
            or ""
        )

        return source, chunk_id, content

    # ============================================================
    # SCORE VECTORIEL
    # ============================================================

    @staticmethod
    def _normalize_relevance_score(score: float) -> float:
        """
        Normalise uniquement pour protéger le Pipeline.

        Le score produit par LangChain/Chroma est conservé dans sa
        sémantique d'origine. On borne simplement les valeurs afin
        d'éviter les valeurs invalides pour le Relevance Gate.
        """

        try:
            score = float(score)
        except (TypeError, ValueError):
            return 0.0

        if score != score:
            return 0.0

        return max(
            0.0,
            min(1.0, score),
        )

    def _vector_similarity_results(
        self,
        query: str,
        k: int,
    ) -> List[Tuple[Document, float]]:

        if k <= 0:
            return []

        raw_results = (
            self.vectordb.similarity_search_with_relevance_scores(
                query,
                k=k,
            )
        )

        results: List[Tuple[Document, float]] = []

        for document, raw_score in raw_results:

            score = self._normalize_relevance_score(
                raw_score
            )

            results.append(
                (
                    document,
                    score,
                )
            )

        return results

    # ============================================================
    # SIMILARITY SCORE THRESHOLD
    # ============================================================

    def _apply_similarity_threshold(
        self,
        results: List[Tuple[Document, float]],
    ) -> List[Tuple[Document, float]]:

        threshold = self.similarity_score_threshold

        if threshold is None:
            return results

        return [
            (document, score)
            for document, score in results
            if score >= threshold
        ]

    # ============================================================
    # BM25
    # ============================================================

    def _bm25_search(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> List[Tuple[Document, float]]:

        if self.bm25 is None:
            raise RuntimeError(
                "Index BM25 non initialisé."
            )

        effective_k = (
            self.k
            if k is None
            else int(k)
        )

        if effective_k <= 0:
            return []

        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: (
                -float(scores[i]),
                i,
            ),
        )

        results: List[Tuple[Document, float]] = []

        for index in ranked_indices:

            score = float(
                scores[index]
            )

            # Important :
            # les scores nuls ne doivent pas produire des documents
            # arbitraires lorsque la requête ne matche aucun terme.
            if score <= 0.0:
                continue

            results.append(
                (
                    self.bm25_documents[index],
                    score,
                )
            )

            if len(results) >= effective_k:
                break

        return results

    # ============================================================
    # HYBRID RRF
    # ============================================================

    def _hybrid_search(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> List[Tuple[Document, float]]:

        if self.bm25 is None:
            raise RuntimeError(
                "Index BM25 non initialisé pour la recherche hybride."
            )

        effective_k = (
            self.k
            if k is None
            else int(k)
        )

        if effective_k <= 0:
            return []

        # On récupère suffisamment de candidats dans les deux systèmes.
        candidate_k = max(
            self.hybrid_k,
            effective_k,
        )

        vector_results = (
            self._vector_similarity_results(
                query=query,
                k=candidate_k,
            )
        )

        bm25_results = (
            self._bm25_search(
                query=query,
                k=candidate_k,
            )
        )

        # Reciprocal Rank Fusion.
        #
        # RRF(d) = somme 1 / (60 + rank)
        #
        # Le score retourné n'est donc PAS un score de similarité
        # compris entre 0 et 1.
        rrf_constant = 60.0

        rrf_scores: Dict[
            Tuple[str, str, str],
            float,
        ] = {}

        document_map: Dict[
            Tuple[str, str, str],
            Document,
        ] = {}

        # --------------------------------------------------------
        # Vectoriel
        # --------------------------------------------------------

        for rank, (
            document,
            _score,
        ) in enumerate(
            vector_results,
            start=1,
        ):

            key = self._document_key(
                document
            )

            document_map[key] = document

            rrf_scores[key] = (
                rrf_scores.get(key, 0.0)
                + 1.0 / (
                    rrf_constant + rank
                )
            )

        # --------------------------------------------------------
        # BM25
        # --------------------------------------------------------

        for rank, (
            document,
            _score,
        ) in enumerate(
            bm25_results,
            start=1,
        ):

            key = self._document_key(
                document
            )

            document_map[key] = document

            rrf_scores[key] = (
                rrf_scores.get(key, 0.0)
                + 1.0 / (
                    rrf_constant + rank
                )
            )

        # --------------------------------------------------------
        # Classement final
        # --------------------------------------------------------

        sorted_keys = sorted(
            rrf_scores.keys(),
            key=lambda key: (
                -rrf_scores[key],
                key,
            ),
        )

        results = [
            (
                document_map[key],
                float(
                    rrf_scores[key]
                ),
            )
            for key in sorted_keys[
                :effective_k
            ]
        ]

        logger.info(
            "Hybrid RRF | query=%r | vector=%d | bm25=%d | fused=%d",
            query,
            len(vector_results),
            len(bm25_results),
            len(results),
        )

        return results

    # ============================================================
    # RETRIEVE
    # ============================================================

    def retrieve(
        self,
        query: str,
    ) -> List[Document]:

        if not query or not query.strip():
            return []

        query = query.strip()

        if self.search_type == "bm25":
            return [
                document
                for document, _score
                in self._bm25_search(query)
            ]

        if self.search_type == "hybrid":
            return [
                document
                for document, _score
                in self._hybrid_search(query)
            ]

        if self.search_type == "similarity_score_threshold":

            results = self._vector_similarity_results(
                query=query,
                k=self.k,
            )

            results = self._apply_similarity_threshold(
                results
            )

            return [
                document
                for document, _score
                in results
            ]

        if self.retriever is None:
            raise RuntimeError(
                "Retriever vectoriel non initialisé."
            )

        return self.retriever.invoke(
            query
        )

    # ============================================================
    # RETRIEVE AVEC SCORES
    # ============================================================

    def retrieve_with_scores(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> List[Tuple[Document, float]]:

        if not query or not query.strip():
            return []

        query = query.strip()

        effective_k = (
            self.k
            if k is None
            else int(k)
        )

        if effective_k <= 0:
            raise ValueError(
                "k doit être > 0."
            )

        # --------------------------------------------------------
        # BM25
        # --------------------------------------------------------

        if self.search_type == "bm25":

            return self._bm25_search(
                query=query,
                k=effective_k,
            )

        # --------------------------------------------------------
        # HYBRID
        # --------------------------------------------------------

        if self.search_type == "hybrid":

            return self._hybrid_search(
                query=query,
                k=effective_k,
            )

        # --------------------------------------------------------
        # SIMILARITY
        # --------------------------------------------------------

        if self.search_type == "similarity":

            return self._vector_similarity_results(
                query=query,
                k=effective_k,
            )

        # --------------------------------------------------------
        # SIMILARITY + THRESHOLD
        # --------------------------------------------------------

        if self.search_type == "similarity_score_threshold":

            results = self._vector_similarity_results(
                query=query,
                k=effective_k,
            )

            return self._apply_similarity_threshold(
                results
            )

        # --------------------------------------------------------
        # MMR
        # --------------------------------------------------------

        if self.search_type == "mmr":

            documents = (
                self.vectordb.max_marginal_relevance_search(
                    query,
                    k=effective_k,
                    fetch_k=max(
                        self.fetch_k,
                        effective_k,
                    ),
                    lambda_mult=self.lambda_mult,
                )
            )

            if not documents:
                return []

            # MMR ne fournit pas directement les scores.
            #
            # On récupère les scores de pertinence des candidats
            # vectoriels afin d'avoir un score traçable pour le Gate.
            candidate_results = (
                self._vector_similarity_results(
                    query=query,
                    k=max(
                        self.fetch_k,
                        effective_k,
                    ),
                )
            )

            score_map = {
                self._document_key(document): float(score)
                for document, score
                in candidate_results
            }

            results: List[
                Tuple[Document, float]
            ] = []

            for document in documents:

                key = self._document_key(
                    document
                )

                score = score_map.get(
                    key,
                    0.0,
                )

                results.append(
                    (
                        document,
                        float(score),
                    )
                )

            return results

        raise RuntimeError(
            f"Type de recherche inconnu : {self.search_type}"
        )

