from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


class Retriever:
    """
    Retriever supportant :

    - similarity
    - mmr
    - similarity_score_threshold

    - bm25

    - hybrid (combinaison vectorielle + BM25 via RRF pur)

    Les recherches vectorielles utilisent Chroma/LangChain.

    La recherche BM25 est lexicale et fonctionne directement
    sur la liste des chunks fournie au constructeur.

    La recherche hybride combine les résultats vectoriels et BM25
    en utilisant l'algorithme RRF (Reciprocal Rank Fusion) pur.

    IMPORTANT :
    Les scores vectoriels et BM25 ne sont pas comparables
    directement. Le Relevance Gate doit donc être calibré
    séparément pour BM25 et hybrid.
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
        documents: Optional[List[Document]] = None,
        k: int = 8,
        search_type: str = "similarity",
        fetch_k: int = 20,
        lambda_mult: float = 0.6,
        hybrid_k: int = 10,
    ):
        if vectordb is None:
            raise ValueError("vectordb ne peut pas être None.")

        if search_type not in self.VALID_SEARCH_TYPES:
            raise ValueError(
                f"search_type='{search_type}' invalide. "
                f"Valeurs autorisées : {sorted(self.VALID_SEARCH_TYPES)}"
            )

        if k <= 0:
            raise ValueError("k doit être > 0.")

        if fetch_k <= 0:
            raise ValueError("fetch_k doit être > 0.")

        if lambda_mult < 0.0 or lambda_mult > 1.0:
            raise ValueError("lambda_mult doit être compris entre 0 et 1.")

        if search_type in ("bm25", "hybrid") and not documents:
            raise ValueError(
                f"La recherche {search_type} nécessite la liste des chunks "
                "dans 'documents'."
            )

        self.vectordb = vectordb
        self.documents = documents or []

        self.k = k
        self.search_type = search_type
        self.fetch_k = fetch_k
        self.lambda_mult = lambda_mult
        self.hybrid_k = hybrid_k

        # BM25
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
        Tokenisation simple adaptée à un corpus technique français.

        On conserve notamment :
        - mots
        - chiffres
        - versions
        - chemins
        - endpoints
        - termes techniques
        """

        if not text:
            return []

        text = text.lower()

        tokens = re.findall(
            r"[a-zàâçéèêëîïôûùüÿñæœ0-9_./:-]+",
            text,
        )

        return tokens

    # ============================================================
    # INITIALISATION BM25
    # ============================================================

    def _initialize_bm25(self) -> None:
        """Construit l'index BM25 à partir des chunks."""

        if BM25Okapi is None:
            raise ImportError(
                "rank-bm25 n'est pas installé.\n"
                "Installe-le avec : pip install rank-bm25"
            )

        self.bm25_documents = list(self.documents)

        tokenized_documents = [
            self._tokenize(doc.page_content)
            for doc in self.bm25_documents
        ]

        self.bm25 = BM25Okapi(tokenized_documents)

    # ============================================================
    # CONSTRUCTION RETRIEVER LANGCHAIN
    # ============================================================

    def _build_retriever(self):
        """
        Construit le retriever vectoriel LangChain.

        BM25 et hybrid sont traités séparément.
        """

        if self.search_type in ("bm25", "hybrid"):
            return None

        if self.search_type == "similarity":
            return self.vectordb.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": self.k,
                },
            )

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
            # Le threshold n'est volontairement PAS appliqué ici.
            # Le Relevance Gate est géré au niveau Pipeline.
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
    # RECHERCHE BM25
    # ============================================================

    def _bm25_search(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Recherche lexicale BM25.

        Retourne :
            [(Document, score), ...]
        """

        if self.bm25 is None:
            raise RuntimeError("Index BM25 non initialisé.")

        k = k or self.k

        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        results: List[Tuple[Document, float]] = []

        for index in ranked_indices[:k]:
            document = self.bm25_documents[index]
            score = float(scores[index])

            results.append((document, score))

        return results

    # ============================================================
    # RECHERCHE HYBRIDE
    # ============================================================

    def _hybrid_search(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Recherche hybride combinant vectorielle et BM25 via RRF pur.

        Utilise l'algorithme RRF (Reciprocal Rank Fusion) pour fusionner
        les résultats des deux méthodes de recherche sans pondération
        hybride (plus de hybrid_alpha).

        Args:
            query: La requête de recherche
            k: Nombre de résultats à retourner

        Returns:
            Liste de tuples (Document, score_rrf)
        """
        if self.bm25 is None:
            raise RuntimeError("Index BM25 non initialisé pour la recherche hybride.")

        k = k or self.k
        hybrid_k = self.hybrid_k or k * 2

        # Récupérer les résultats vectoriels
        vectorial_results = self.vectordb.similarity_search_with_relevance_scores(
            query,
            k=hybrid_k,
        )

        # Récupérer les résultats BM25
        bm25_results = self._bm25_search(query, k=hybrid_k)

        # Fusion RRF (Reciprocal Rank Fusion)
        rrf_scores: Dict[Tuple[str, str], float] = {}
        document_map: Dict[Tuple[str, str], Document] = {}

        # Constante RRF (typiquement 60)
        rrf_constant = 60

        # Traiter les résultats vectoriels
        for rank, (doc, score) in enumerate(vectorial_results, 1):
            source = str(doc.metadata.get("source", ""))
            key = (source, doc.page_content)
            document_map[key] = doc

            # Score RRF: 1 / (rrf_constant + rank)
            rrf_scores[key] = rrf_scores.get(key, 0) + (1 / (rrf_constant + rank))

        # Traiter les résultats BM25
        for rank, (doc, score) in enumerate(bm25_results, 1):
            source = str(doc.metadata.get("source", ""))
            key = (source, doc.page_content)
            document_map[key] = doc

            # Score RRF: 1 / (rrf_constant + rank)
            rrf_scores[key] = rrf_scores.get(key, 0) + (1 / (rrf_constant + rank))

        # Trier par score RRF combiné
        sorted_results = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        # Retourner les k meilleurs résultats
        results: List[Tuple[Document, float]] = []
        for (source, content), rrf_score in sorted_results[:k]:
            doc = document_map.get((source, content))
            if doc:
                results.append((doc, rrf_score))

        return results

    # ============================================================
    # RETRIEVE
    # ============================================================

    def retrieve(self, query: str) -> List[Document]:
        """
        Recherche les documents pertinents.
        """

        if not query or not query.strip():
            return []

        query = query.strip()

        if self.search_type == "bm25":
            results = self._bm25_search(query)

            documents = [doc for doc, _ in results]

        elif self.search_type == "hybrid":
            results = self._hybrid_search(query)

            documents = [doc for doc, _ in results]

        else:
            documents = self.retriever.invoke(query)

        return documents

    # ============================================================
    # RETRIEVE AVEC SCORES
    # ============================================================

    def retrieve_with_scores(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Recherche avec scores.

        Retour :
            [(Document, score), ...]
        """

        if not query or not query.strip():
            return []

        query = query.strip()
        effective_k = self.k if k is None else int(k)
        if effective_k <= 0:
            raise ValueError("k doit être > 0.")

        # --------------------------------------------------------
        # BM25
        # --------------------------------------------------------

        if self.search_type == "bm25":
            return self._bm25_search(query, k=effective_k)

        # --------------------------------------------------------
        # Hybrid
        # --------------------------------------------------------

        if self.search_type == "hybrid":
            return self._hybrid_search(query, k=effective_k)

        # --------------------------------------------------------
        # Similarity
        # --------------------------------------------------------

        if self.search_type == "similarity":

            results = self.vectordb.similarity_search_with_relevance_scores(
                query,
                k=effective_k,
            )

            return [
                (doc, float(score))
                for doc, score in results
            ]

        # --------------------------------------------------------
        # Similarity + Threshold
        # --------------------------------------------------------

        if self.search_type == "similarity_score_threshold":

            results = self.vectordb.similarity_search_with_relevance_scores(
                query,
                k=effective_k,
            )

            return [
                (doc, float(score))
                for doc, score in results
            ]

        # --------------------------------------------------------
        # MMR
        # --------------------------------------------------------

        if self.search_type == "mmr":

            documents = self.vectordb.max_marginal_relevance_search(
                query,
                k=effective_k,
                fetch_k=self.fetch_k,
                lambda_mult=self.lambda_mult,
            )

            if not documents:
                return []

            # MMR ne retourne pas directement les scores.
            # On récupère les scores vectoriels des documents sélectionnés.
            candidate_results = (
                self.vectordb.similarity_search_with_relevance_scores(
                    query,
                    k=max(self.fetch_k, effective_k),
                )
            )

            score_map: Dict[Tuple[str, str], float] = {}

            for doc, score in candidate_results:

                source = str(
                    doc.metadata.get("source", "")
                )

                content = doc.page_content

                key = (
                    source,
                    content,
                )

                score_map[key] = float(score)

            results: List[Tuple[Document, float]] = []

            for doc in documents:

                source = str(
                    doc.metadata.get("source", "")
                )

                key = (
                    source,
                    doc.page_content,
                )

                score = score_map.get(key, 0.0)

                results.append(
                    (doc, score)
                )

            return results

        raise RuntimeError(
            f"Type de recherche inconnu : {self.search_type}"
        )