from __future__ import annotations

import time
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_ollama import ChatOllama

from .document_loader import DocumentLoader
from .embedder import Embedder
from .chunking_agent import ChunkingAgent
from .chroma_store import ChromaStore
from .retriever import Retriever
from .generator import Generator
from .conversation_history import ConversationHistory
from .query_rewriter import QueryRewriter


class RAGPipeline:
    """
    Pipeline RAG TELNET SmartConnect.

    Architecture :

        Question
            ↓
        ConversationHistory
            ↓
        QueryRewriter
            ↓
        Search Query
            ↓
        Retriever (hybrid avec RRF pur)
            ↓
        Relevance Gate
            ↓
        Deduplication
            ↓
        Context
            ↓
        Generator / Mistral
            ↓
        Answer
            ↓
        ConversationHistory
    """

    def __init__(
        self,
        data_dir: str = "./data",
        db_dir: str = "./chroma_db",
        collection_name: str = "telnet_support",

        embedding_model: str = "BAAI/bge-m3",
        llm_model: str = "mistral",

        retrieval_type: str = "mmr",
        retrieval_k: int = 8,
        retrieval_fetch_k: int = 20,
        retrieval_lambda: float = 0.6,
        hybrid_k: int = 10,

        relevance_threshold: Optional[float] = None,
        bm25_relevance_threshold: Optional[float] = None,
        hybrid_relevance_threshold: Optional[float] = None,

        max_context_documents: int = 6,

        use_history: bool = True,
        max_history: int = 10,
        max_history_chars: int = 5000,

        enable_query_rewriting: bool = True,
    ) -> None:

        # ========================================================
        # CONFIGURATION
        # ========================================================

        self.data_dir = data_dir
        self.db_dir = db_dir
        self.collection_name = collection_name

        self.embedding_model = embedding_model
        self.llm_model = llm_model

        self.retrieval_type = retrieval_type
        self.retrieval_k = retrieval_k
        self.retrieval_fetch_k = retrieval_fetch_k
        self.retrieval_lambda = retrieval_lambda
        self.hybrid_k = hybrid_k

        self.relevance_threshold = relevance_threshold
        self.bm25_relevance_threshold = bm25_relevance_threshold
        self.hybrid_relevance_threshold = hybrid_relevance_threshold
        self.chunking_version = "chunking_agent_v1"
        self.index_documents: List[Document] = []

        if relevance_threshold is not None and relevance_threshold < 0:
            raise ValueError("relevance_threshold doit être >= 0 ou None.")
        if bm25_relevance_threshold is not None and bm25_relevance_threshold < 0:
            raise ValueError("bm25_relevance_threshold doit être >= 0 ou None.")
        if hybrid_relevance_threshold is not None and hybrid_relevance_threshold < 0:
            raise ValueError("hybrid_relevance_threshold doit être >= 0 ou None.")

        self.max_context_documents = max_context_documents

        self.use_history = use_history
        self.max_history = max_history
        self.max_history_chars = max_history_chars

        self.enable_query_rewriting = (
            enable_query_rewriting
        )

        # ========================================================
        # DOCUMENT LOADER
        # ========================================================

        self.loader = DocumentLoader(
            data_dir=data_dir
        )

        # ========================================================
        # EMBEDDER
        # ========================================================

        self.embedder = Embedder(
            model_name=embedding_model
        )

        self.embeddings = (
            self.embedder.get_embeddings()
        )

        # ========================================================
        # CHUNKER
        # ========================================================

        self.chunker = ChunkingAgent(
            model_name=llm_model,
            embeddings=self.embeddings,
            enable_llm_analysis=True,
            target_chunk_size=900,
            min_chunk_size=300,
            max_chunk_size=1200,
        )

        # ========================================================
        # CHROMA STORE
        # ========================================================

        self.vector_store = ChromaStore(
            persist_directory=db_dir,
            collection_name=collection_name,
        )

        # ========================================================
        # COMPONENTS
        # ========================================================

        self.retriever: Optional[Retriever] = None
        self.generator: Optional[Generator] = None

        # ========================================================
        # HISTORY
        # ========================================================

        self.history = ConversationHistory(
            max_history=max_history,
            max_context_chars=max_history_chars,
        )

        # ========================================================
        # QUERY REWRITER
        # ========================================================

        self.query_rewriter: Optional[
            QueryRewriter
        ] = None

        if (
            self.use_history
            and self.enable_query_rewriting
        ):

            self.query_rewriter = QueryRewriter(
                model_name=llm_model,
                temperature=0.0,
                max_history_chars=max_history_chars,
            )

        # ========================================================
        # QUESTION UNDERSTANDING
        # ========================================================
        # Petit appel LLM indépendant qui vérifie si la question
        # est compréhensible AVANT le QueryRewriter.
        self.question_validator = ChatOllama(
            model=llm_model,
            temperature=0.0,
            num_predict=128,
        )

    # ============================================================
    # INITIALISATION
    # ============================================================

    def _initialize_query_components(self) -> None:
        """
        Initialise Retriever et Generator.
        """

        vectorstore = getattr(
            self.vector_store,
            "vectorstore",
            None,
        )

        if vectorstore is None:
            raise RuntimeError(
                "La base vectorielle n'est pas chargée. "
                "Appelez load_existing() ou build_index()."
            )

        # ========================================================
        # RETRIEVER
        # ========================================================

        retriever_documents = (
            self.index_documents
            if self.retrieval_type in ("bm25", "hybrid")
            else None
        )

        self.retriever = Retriever(
            vectordb=vectorstore,
            documents=retriever_documents,
            search_type=self.retrieval_type,
            k=self.retrieval_k,
            fetch_k=self.retrieval_fetch_k,
            lambda_mult=self.retrieval_lambda,
            hybrid_k=self.hybrid_k,
        )

        # ========================================================
        # GENERATOR
        # ========================================================

        self.generator = Generator(
            model_name=self.llm_model,
            temperature=0.0,
            num_predict=512,
        )

    # ============================================================
    # BUILD INDEX
    # ============================================================

    def build_index(self) -> Dict[str, Any]:

        # --------------------------------------------------------
        # LOAD DOCUMENTS
        # --------------------------------------------------------

        documents = self.loader.load()

        if not documents:
            raise ValueError(
                "Aucun document trouvé."
            )

        # --------------------------------------------------------
        # CHUNKING
        # --------------------------------------------------------

        chunks = (
            self.chunker.chunk_documents(
                documents
            )
        )

        if not chunks:
            raise ValueError(
                "Aucun chunk généré."
            )

        # --------------------------------------------------------
        # CHROMA
        # --------------------------------------------------------

        self.vector_store.create_or_replace(
            chunks=chunks,
            embeddings=self.embeddings,
            documents=documents,
            embedding_model=self.embedding_model,
            chunking_version=self.chunking_version,
        )
        self.index_documents = list(chunks)

        # --------------------------------------------------------
        # QUERY COMPONENTS
        # --------------------------------------------------------

        self._initialize_query_components()

        return {
            "documents": len(documents),
            "chunks": len(chunks),
        }

    # ============================================================
    # LOAD EXISTING
    # ============================================================

    def load_existing(self, strict_compatibility: bool = True) -> None:
        """Charge un index persistant et reconstruit les données auxiliaires nécessaires."""

        if strict_compatibility and not self.vector_store.is_compatible(
            embedding_model=self.embedding_model,
            chunking_version=self.chunking_version,
        ):
            manifest = self.vector_store.get_manifest()
            if manifest:
                raise ValueError(
                    "Index Chroma incompatible avec la configuration actuelle : "
                    f"embedding={manifest.get('embedding_model')!r}, "
                    f"chunking={manifest.get('chunking_version')!r}."
                )
            if strict_compatibility:
                raise ValueError(
                    "Manifest Chroma absent. Utilisez strict_compatibility=False "
                    "ou reconstruisez l'index."
                )

        self.vector_store.load(embeddings=self.embeddings)
        self.index_documents = self.vector_store.get_documents()
        if not self.index_documents:
            raise ValueError("Aucun chunk récupérable depuis l'index Chroma.")

        self._initialize_query_components()

    # ============================================================
    # QUESTION UNDERSTANDING
    # ============================================================

    def _check_question_understandability(
        self,
        question: str,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Détermine avec le LLM si la question est suffisamment
        compréhensible pour être traitée par le RAG.

        Important :
        - cette étape intervient AVANT le QueryRewriter ;
        - elle ne cherche pas la réponse ;
        - elle vérifie uniquement si l'intention de l'utilisateur
          peut être comprise ;
        - une question courte mais claire ("JWT ?", "API ?")
          peut être considérée comme compréhensible.
        """

        original_question = (
            question or ""
        ).strip()

        if not original_question:
            return (
                False,
                {
                    "understandable": False,
                    "reason": "empty_question",
                },
            )

        prompt = f"""
Tu es un classificateur de questions pour un chatbot de support
TELNET SmartConnect.

Ta seule tâche est de déterminer si la question de l'utilisateur
est COMPRÉHENSIBLE, c'est-à-dire si son intention peut être
identifiée suffisamment clairement pour lancer une recherche
documentaire.

Une question peut être courte et quand même être compréhensible.
Exemples :
- "JWT ?" -> compréhensible
- "API ?" -> compréhensible
- "Comment obtenir un token JWT ?" -> compréhensible
- "Comment accéder aux données d'un device ?" -> compréhensible

Une question est NON compréhensible si elle est manifestement
aléatoire, vide de sens, composée de caractères sans intention
identifiable ou trop ambiguë pour savoir ce que l'utilisateur
demande.
Exemples :
- "jnkjl" -> non compréhensible
- "vdvds" -> non compréhensible
- "asdfgh" -> non compréhensible

Ne juge pas si la question est vraie ou fausse.
Ne cherche pas à répondre à la question.
Ne la réécris pas.
Juge uniquement sa compréhensibilité.

Réponds UNIQUEMENT avec un JSON valide de cette forme :
{{
  "understandable": true ou false,
  "reason": "courte explication"
}}

Question utilisateur :
{original_question}
"""

        start = time.perf_counter()

        try:
            response = self.question_validator.invoke(prompt)

            content = getattr(
                response,
                "content",
                response,
            )

            if isinstance(content, list):
                content = "".join(
                    str(item)
                    for item in content
                )

            content = str(content).strip()

            # Nettoyage minimal si le modèle entoure le JSON
            # avec ```json ... ```.
            content = re.sub(
                r"^```(?:json)?\s*",
                "",
                content,
                flags=re.IGNORECASE,
            )
            content = re.sub(
                r"\s*```$",
                "",
                content,
            ).strip()

            import json

            result = json.loads(content)

            understandable = result.get(
                "understandable"
            )

            if isinstance(
                understandable,
                str,
            ):
                understandable = (
                    understandable.lower()
                    in {
                        "true",
                        "1",
                        "yes",
                        "oui",
                    }
                )

            understandable = bool(
                understandable
            )

            return (
                understandable,
                {
                    "understandable": understandable,
                    "reason": str(
                        result.get(
                            "reason",
                            "",
                        )
                    ),
                    "latency": (
                        time.perf_counter()
                        - start
                    ),
                },
            )

        except Exception as exc:
            # En cas d'échec du classificateur, on ne bloque pas
            # une vraie question : on laisse le pipeline continuer.
            return (
                True,
                {
                    "understandable": True,
                    "reason": "validator_error_fallback",
                    "error": str(exc),
                    "latency": (
                        time.perf_counter()
                        - start
                    ),
                },
            )

    # ============================================================
    # QUERY REWRITING
    # ============================================================

    def _rewrite_query(
        self,
        question: str,
    ) -> Tuple[str, Dict[str, Any]]:

        original_question = (
            question or ""
        ).strip()

        # --------------------------------------------------------
        # DISABLED
        # --------------------------------------------------------

        if not self.enable_query_rewriting:

            return (
                original_question,
                {
                    "enabled": False,
                    "rewritten": False,
                    "original_query": original_question,
                    "search_query": original_question,
                    "reason": "disabled",
                },
            )

        # --------------------------------------------------------
        # REWRITER UNAVAILABLE
        # --------------------------------------------------------

        if self.query_rewriter is None:

            return (
                original_question,
                {
                    "enabled": False,
                    "rewritten": False,
                    "original_query": original_question,
                    "search_query": original_question,
                    "reason": "rewriter_unavailable",
                },
            )

        # --------------------------------------------------------
        # HISTORY DISABLED
        # --------------------------------------------------------

        if not self.use_history:

            return (
                original_question,
                {
                    "enabled": True,
                    "rewritten": False,
                    "original_query": original_question,
                    "search_query": original_question,
                    "history_used": False,
                    "reason": "history_disabled",
                },
            )

        # --------------------------------------------------------
        # HISTORY
        # --------------------------------------------------------

        history_for_rewriter = (
            self.history.get_context_for_rewriting(
                max_chars=4000,
                max_messages=6,
            )
        )

        if not history_for_rewriter:

            return (
                original_question,
                {
                    "enabled": True,
                    "rewritten": False,
                    "original_query": original_question,
                    "search_query": original_question,
                    "history_used": False,
                    "reason": "no_history",
                },
            )

        # --------------------------------------------------------
        # REWRITE
        # --------------------------------------------------------

        start = time.perf_counter()

        try:

            rewritten_query = (
                self.query_rewriter.rewrite(
                    question=original_question,
                    history=history_for_rewriter,
                )
            )

        except Exception as exc:

            elapsed = (
                time.perf_counter()
                - start
            )

            return (
                original_question,
                {
                    "enabled": True,
                    "rewritten": False,
                    "original_query": original_question,
                    "search_query": original_question,
                    "history_used": True,
                    "rewriter_latency": elapsed,
                    "reason": "rewriter_error",
                    "error": str(exc),
                },
            )

        elapsed = (
            time.perf_counter()
            - start
        )

        rewritten_query = (
            rewritten_query or ""
        ).strip()

        if not rewritten_query:
            rewritten_query = original_question

        was_rewritten = (
            rewritten_query.lower()
            != original_question.lower()
        )

        metadata = {
            "enabled": True,
            "rewritten": was_rewritten,
            "original_query": original_question,
            "search_query": rewritten_query,
            "history_used": True,
            "rewriter_latency": elapsed,
        }

        return (
            rewritten_query,
            metadata,
        )

    # ============================================================
    # DEDUPLICATION
    # ============================================================

    @staticmethod
    def _deduplicate_documents(
        documents: List[Document],
    ) -> List[Document]:

        seen = set()
        unique_documents = []

        for document in documents:

            content = (
                document.page_content or ""
            ).strip()

            metadata = (
                document.metadata or {}
            )

            source = (
                metadata.get("source")
                or metadata.get("file_path")
                or metadata.get("filename")
                or ""
            )

            chunk_id = metadata.get(
                "chunk_id"
            )

            key = (
                source,
                chunk_id,
                content,
            )

            if key in seen:
                continue

            seen.add(key)

            unique_documents.append(
                document
            )

        return unique_documents

    # ============================================================
    # RELEVANCE GATE
    # ============================================================

    def _apply_relevance_gate(
        self,
        results: List[Tuple[Document, float]],
    ) -> Tuple[
        List[Tuple[Document, float]],
        Dict[str, Any],
    ]:

        threshold = (
            self.bm25_relevance_threshold
            if self.retrieval_type == "bm25"
            else self.hybrid_relevance_threshold
            if self.retrieval_type == "hybrid"
            else self.relevance_threshold
        )

        if not results:
            return (
                [],
                {
                    "input_count": 0,
                    "accepted_count": 0,
                    "rejected_count": 0,
                    "threshold": threshold,
                    "applied": threshold is not None,
                    "score_type": (
                        "bm25"
                        if self.retrieval_type == "bm25"
                        else "hybrid"
                        if self.retrieval_type == "hybrid"
                        else "relevance"
                    ),
                },
            )

        relevant = []
        rejected = []

        for document, score in results:
            score = float(score)

            if threshold is None or score >= threshold:

                relevant.append(
                    (
                        document,
                        score,
                    )
                )

            else:

                rejected.append(
                    (
                        document,
                        score,
                    )
                )

        metadata = {
            "input_count": len(results),
            "accepted_count": len(relevant),
            "rejected_count": len(rejected),
            "threshold": threshold,
            "applied": threshold is not None,
            "score_type": (
                "bm25"
                if self.retrieval_type == "bm25"
                else "hybrid"
                if self.retrieval_type == "hybrid"
                else "relevance"
            ),
        }

        return (
            relevant,
            metadata,
        )

    def _score_is_accepted(self, score: float) -> bool:
        """
        Check if a score meets the relevance threshold.

        Args:
            score: The relevance score to check

        Returns:
            True if the score is accepted, False otherwise
        """
        threshold = (
            self.bm25_relevance_threshold
            if self.retrieval_type == "bm25"
            else self.hybrid_relevance_threshold
            if self.retrieval_type == "hybrid"
            else self.relevance_threshold
        )

        return threshold is None or score >= threshold

    # ============================================================
    # ASK
    # ============================================================

    def ask(
        self,
        question: str,
    ) -> Dict[str, Any]:

        question = (
            question or ""
        ).strip()

        if not question:

            raise ValueError(
                "La question ne peut pas être vide."
            )

        # ========================================================
        # INITIALISATION
        # ========================================================

        if (
            self.retriever is None
            or self.generator is None
        ):

            self._initialize_query_components()

        # ========================================================
        # HISTORY
        # ========================================================

        if self.use_history:

            history_text = (
                self.history.get_context_for_prompt(
                    max_chars=self.max_history_chars
                )
            )

        else:

            history_text = ""

        # ========================================================
        # QUESTION UNDERSTANDING
        # ========================================================
        # Cette vérification est volontairement placée AVANT
        # le QueryRewriter.
        (
            question_understandable,
            understanding_metadata,
        ) = self._check_question_understandability(
            question
        )

        if not question_understandable:

            answer = (
                "Je n'ai pas compris votre question. "
                "Veuillez reformuler votre demande concernant "
                "SmartConnect."
            )

            if self.use_history:
                self.history.add_user_message(
                    question
                )
                self.history.add_assistant_message(
                    answer,
                    sources=[],
                )

            return {
                "answer": answer,
                "question": question,
                "search_query": question,
                "documents": [],
                "scores": [],
                "source_documents": [],
                "retrieval_scores": [],
                "retrieved_count": 0,
                "relevant_count": 0,
                "relevance_threshold": (
                    self.relevance_threshold
                ),
                "relevance_gate": {
                    "input_count": 0,
                    "accepted_count": 0,
                    "rejected_count": 0,
                    "threshold": (
                        self.relevance_threshold
                    ),
                    "applied": False,
                    "score_type": "not_retrieved",
                },
                "question_understanding": (
                    understanding_metadata
                ),
                "query_rewrite": {
                    "enabled": bool(
                        self.enable_query_rewriting
                    ),
                    "rewritten": False,
                    "original_query": question,
                    "search_query": question,
                    "reason": (
                        "question_not_understandable"
                    ),
                },
            }

        # ========================================================
        # QUERY REWRITING
        # ========================================================

        (
            search_query,
            _rewrite_metadata,
        ) = self._rewrite_query(
            question
        )

        # ========================================================
        # RETRIEVAL
        # ========================================================

        retrieval_results = (
            self.retriever.retrieve_with_scores(
                query=search_query,
                k=self.retrieval_k,
            )
        )

        # ========================================================
        # RELEVANCE GATE
        # ========================================================

        (
            relevant_results,
            _gate_metadata,
        ) = self._apply_relevance_gate(
            retrieval_results
        )

        # ========================================================
        # DEDUPLICATION
        # ========================================================

        seen = set()
        documents_with_scores = []

        for document, score in relevant_results:
            content = (document.page_content or "").strip()
            metadata = document.metadata or {}
            source = (
                metadata.get("source")
                or metadata.get("relative_path")
                or metadata.get("filename")
                or ""
            )
            chunk_id = metadata.get("chunk_id")
            key = (source, chunk_id, content)
            if key in seen:
                continue
            seen.add(key)
            document.metadata["retrieval_score"] = float(score)
            document.metadata["retrieval_score_type"] = (
                "bm25"
                if self.retrieval_type == "bm25"
                else "hybrid"
                if self.retrieval_type == "hybrid"
                else "vector_relevance"
            )
            documents_with_scores.append((document, float(score)))

        # ========================================================
        # CONTEXT LIMIT
        # ========================================================

        documents_with_scores = (
            documents_with_scores[
                :self.max_context_documents
            ]
        )

        relevant_documents = [
            document
            for document, _score
            in documents_with_scores
        ]

        relevant_scores = [
            float(score)
            for _document, score
            in documents_with_scores
        ]

        # ========================================================
        # GENERATION
        # ========================================================

        if not relevant_documents:

            answer = (
                "Je ne dispose pas d'informations "
                "suffisamment pertinentes dans la "
                "documentation disponible pour répondre "
                "à cette question."
            )

        else:

            generator_result = (
                self.generator.generate(
                    question=question,
                    standalone_question=search_query,
                    documents=relevant_documents,
                    history=history_text,
                )
            )

            if isinstance(
                generator_result,
                dict,
            ):

                answer = generator_result.get(
                    "answer",
                    generator_result.get(
                        "result",
                        "",
                    ),
                )

            else:

                answer = str(
                    generator_result
                )

        answer = (
            answer or ""
        ).strip()

        # ========================================================
        # SAVE HISTORY
        # ========================================================

        source_metadata = []

        for document in relevant_documents:

            metadata = (
                document.metadata or {}
            )

            source_metadata.append(
                {
                    "source": metadata.get(
                        "source"
                    ),

                    "page": metadata.get(
                        "page"
                    ),

                    "chunk_id": metadata.get(
                        "chunk_id"
                    ),
                }
            )

        if self.use_history:

            self.history.add_user_message(
                question
            )

            self.history.add_assistant_message(
                answer,
                sources=source_metadata,
            )

        # ========================================================
        # RESULT
        # ========================================================

        return {

            "answer": answer,

            "question": question,

            "search_query": search_query,

            "documents": relevant_documents,

            "scores": relevant_scores,

            # Compatibilité avec main.py
            "source_documents": (
                relevant_documents
            ),

            "retrieval_scores": (
                relevant_scores
            ),

            "retrieved_count": len(
                retrieval_results
            ),

            "relevant_count": len(
                relevant_documents
            ),
            "question_understanding": (
                understanding_metadata
            ),
        }

    # ============================================================
    # HISTORY
    # ============================================================

    def get_history(
        self,
    ) -> List[Dict[str, Any]]:

        return self.history.get_history()

    def clear_history(self) -> None:

        self.history.clear()

    def get_history_text(
        self,
        max_chars: int = 5000,
    ) -> str:

        return (
            self.history.get_context_for_prompt(
                max_chars=max_chars
            )
        )

    # ============================================================
    # STATUS
    # ============================================================

    def status(self) -> Dict[str, Any]:

        return {

            "data_dir": self.data_dir,

            "db_dir": self.db_dir,

            "collection_name": (
                self.collection_name
            ),

            "embedding_model": (
                self.embedding_model
            ),

            "llm_model": self.llm_model,

            "retrieval_type": (
                self.retrieval_type
            ),

            "retrieval_k": (
                self.retrieval_k
            ),

            "retrieval_fetch_k": (
                self.retrieval_fetch_k
            ),

            "retrieval_lambda": (
                self.retrieval_lambda
            ),

            "relevance_threshold": self.relevance_threshold,

            "bm25_relevance_threshold": self.bm25_relevance_threshold,

            "hybrid_relevance_threshold": self.hybrid_relevance_threshold,

            "chunking_version": self.chunking_version,

            "history_enabled": (
                self.use_history
            ),

            "history_messages": len(
                self.history
            ),

            "query_rewriting": (
                self.enable_query_rewriting
            ),
            "question_understanding": True,

            "retriever_initialized": (
                self.retriever is not None
            ),

            "generator_initialized": (
                self.generator is not None
            ),
        }