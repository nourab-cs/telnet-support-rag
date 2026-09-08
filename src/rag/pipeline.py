
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

from .document_loader import DocumentLoader
from .embedder import Embedder
from .chunking_agent import ChunkingAgent
from .chroma_store import ChromaStore
from .retriever import Retriever
from .generator import Generator
from .conversation_history import ConversationHistory
from .query_rewriter import QueryRewriter


logger = logging.getLogger(__name__)


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
        Retriever
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

        relevance_threshold: float = 0.40,

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

        self.relevance_threshold = (
            relevance_threshold
        )

        self.max_context_documents = (
            max_context_documents
        )

        self.use_history = use_history
        self.max_history = max_history
        self.max_history_chars = (
            max_history_chars
        )

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

        # IMPORTANT :
        # Embedder expose get_embeddings()
        # et non .embeddings

        self.embeddings = (
            self.embedder.get_embeddings()
        )

        logger.info(
            "Embeddings initialisés | model=%s",
            embedding_model,
        )

        # ========================================================
        # CHUNKER
        # ========================================================

        self.chunker = ChunkingAgent(
            model_name=llm_model,
            embeddings=self.embeddings,
            enable_llm_analysis=False,
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

        logger.info(
            "RAGPipeline initialisé | "
            "collection=%s | retrieval=%s | "
            "k=%d | fetch_k=%d | lambda=%.2f | "
            "threshold=%.2f | history=%s | "
            "query_rewriting=%s",
            collection_name,
            retrieval_type,
            retrieval_k,
            retrieval_fetch_k,
            retrieval_lambda,
            relevance_threshold,
            use_history,
            enable_query_rewriting,
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

        self.retriever = Retriever(
            vectordb=vectorstore,
            search_type=self.retrieval_type,
            k=self.retrieval_k,
            fetch_k=self.retrieval_fetch_k,
            lambda_mult=self.retrieval_lambda,
            score_threshold=self.relevance_threshold,
            enable_tracing=True,
        )

        # ========================================================
        # GENERATOR
        # ========================================================

        self.generator = Generator(
            model_name=self.llm_model,
            temperature=0.0,
            num_predict=512,
        )

        logger.info(
            "Composants query initialisés | "
            "retriever=%s | generator=%s",
            type(self.retriever).__name__,
            type(self.generator).__name__,
        )

    # ============================================================
    # BUILD INDEX
    # ============================================================

    def build_index(self) -> Dict[str, Any]:

        start = time.perf_counter()

        logger.info(
            "Début construction index."
        )

        # --------------------------------------------------------
        # LOAD DOCUMENTS
        # --------------------------------------------------------

        documents = (
            self.loader.load_documents()
        )

        if not documents:

            raise ValueError(
                "Aucun document trouvé."
            )

        logger.info(
            "Documents chargés | count=%d",
            len(documents),
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

        logger.info(
            "Chunks générés | count=%d",
            len(chunks),
        )

        # --------------------------------------------------------
        # CHROMA
        # --------------------------------------------------------

        self.vector_store.create_or_replace(
            chunks=chunks,
            embeddings=self.embeddings,
            documents=documents,
            embedding_model=self.embedding_model,
            chunking_version="chunking_agent_v1",
        )

        # --------------------------------------------------------
        # QUERY COMPONENTS
        # --------------------------------------------------------

        self._initialize_query_components()

        elapsed = (
            time.perf_counter()
            - start
        )

        logger.info(
            "Index construit | "
            "documents=%d | chunks=%d | latency=%.3fs",
            len(documents),
            len(chunks),
            elapsed,
        )

        return {
            "documents": len(documents),
            "chunks": len(chunks),
            "latency": elapsed,
        }

    # ============================================================
    # LOAD EXISTING
    # ============================================================

    def load_existing(self) -> None:

        logger.info(
            "Chargement de la base vectorielle."
        )

        self.vector_store.load(
            embeddings=self.embeddings
        )

        self._initialize_query_components()

        logger.info(
            "Base vectorielle chargée avec succès."
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

            logger.exception(
                "Erreur QueryRewriter."
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

        logger.info(
            "Query rewriting terminé | "
            "rewritten=%s | latency=%.3fs | "
            "original=%r | search=%r",
            was_rewritten,
            elapsed,
            original_question,
            rewritten_query,
        )

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

        if not results:

            return (
                [],
                {
                    "input_count": 0,
                    "accepted_count": 0,
                    "rejected_count": 0,
                    "threshold": self.relevance_threshold,
                },
            )

        relevant = []
        rejected = []

        for document, score in results:

            score = float(score)

            if score >= self.relevance_threshold:

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
            "threshold": self.relevance_threshold,
        }

        logger.info(
            "Relevance Gate | "
            "input=%d | accepted=%d | "
            "rejected=%d | threshold=%.3f",
            len(results),
            len(relevant),
            len(rejected),
            self.relevance_threshold,
        )

        return (
            relevant,
            metadata,
        )

    # ============================================================
    # ASK
    # ============================================================

    def ask(
        self,
        question: str,
    ) -> Dict[str, Any]:

        start_total = (
            time.perf_counter()
        )

        question = (
            question or ""
        ).strip()

        if not question:

            raise ValueError(
                "La question ne peut pas être vide."
            )

        logger.info(
            "=" * 70
        )

        logger.info(
            "Nouvelle question | question=%r",
            question,
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
        # QUERY REWRITING
        # ========================================================

        (
            search_query,
            rewrite_trace,
        ) = self._rewrite_query(
            question
        )

        # ========================================================
        # RETRIEVAL
        # ========================================================

        retrieval_start = (
            time.perf_counter()
        )

        retrieval_results = (
            self.retriever.retrieve_with_scores(
                query=search_query,
                k=self.retrieval_k,
            )
        )

        retrieval_latency = (
            time.perf_counter()
            - retrieval_start
        )

        logger.info(
            "Retrieval terminé | "
            "query=%r | results=%d | "
            "latency=%.3fs",
            search_query,
            len(retrieval_results),
            retrieval_latency,
        )

        # ========================================================
        # RELEVANCE GATE
        # ========================================================

        (
            relevant_results,
            gate_trace,
        ) = self._apply_relevance_gate(
            retrieval_results
        )

        # ========================================================
        # DEDUPLICATION
        # ========================================================

        seen = set()

        documents_with_scores = []

        for document, score in relevant_results:

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

            # Ajouter le score dans les metadata
            # pour que Generator puisse l'afficher.
            document.metadata[
                "retrieval_score"
            ] = float(score)

            documents_with_scores.append(
                (
                    document,
                    float(score),
                )
            )

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

        logger.info(
            "Contexte final | documents=%d",
            len(relevant_documents),
        )

        # ========================================================
        # TRACE DOCUMENTS
        # ========================================================

        documents_trace = []

        for rank, (
            document,
            score,
        ) in enumerate(
            retrieval_results,
            start=1,
        ):

            metadata = (
                document.metadata or {}
            )

            documents_trace.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "source": metadata.get(
                        "source"
                    ),
                    "file": metadata.get(
                        "file"
                    ),
                    "filename": metadata.get(
                        "filename"
                    ),
                    "page": metadata.get(
                        "page"
                    ),
                    "chunk_id": metadata.get(
                        "chunk_id"
                    ),
                    "accepted": (
                        float(score)
                        >= self.relevance_threshold
                    ),
                    "preview": (
                        (
                            document.page_content
                            or ""
                        )[:300]
                        .replace(
                            "\n",
                            " ",
                        )
                    ),
                }
            )

        # ========================================================
        # GENERATION
        # ========================================================

        generation_start = (
            time.perf_counter()
        )

        if not relevant_documents:

            answer = (
                "Je ne dispose pas d'informations "
                "suffisamment pertinentes dans la "
                "documentation disponible pour répondre "
                "à cette question."
            )

            generation_trace = {
                "executed": False,
                "reason": "no_relevant_documents",
            }

        else:

            generator_result = (
                self.generator.generate(
                    question=question,
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

                generation_trace = {
                    "executed": True,
                    "validation_passed": (
                        generator_result.get(
                            "validation_passed",
                            True,
                        )
                    ),
                }

            else:

                answer = str(
                    generator_result
                )

                generation_trace = {
                    "executed": True,
                    "validation_passed": True,
                }

        generation_latency = (
            time.perf_counter()
            - generation_start
        )

        answer = (
            answer or ""
        ).strip()

        logger.info(
            "Generation terminée | latency=%.3fs",
            generation_latency,
        )

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
        # TOTAL LATENCY
        # ========================================================

        total_latency = (
            time.perf_counter()
            - start_total
        )

        # ========================================================
        # TRACE
        # ========================================================

        trace = {

            "original_question": question,

            "query_rewriting": rewrite_trace,

            "retrieval": {

                "configured_method": (
                    self.retrieval_type
                ),

                "k": self.retrieval_k,

                "fetch_k": (
                    self.retrieval_fetch_k
                ),

                "lambda": (
                    self.retrieval_lambda
                ),

                "query": search_query,

                "latency": retrieval_latency,

                "result_count": len(
                    retrieval_results
                ),
            },

            "relevance_gate": gate_trace,

            "documents": documents_trace,

            "context": {

                "selected_count": len(
                    relevant_documents
                ),

                "max_context_documents": (
                    self.max_context_documents
                ),

                "scores": relevant_scores,
            },

            "generation": {

                "model": self.llm_model,

                "latency": generation_latency,

                **generation_trace,
            },

            "history": {

                "enabled": self.use_history,

                "messages": len(
                    self.history
                ),
            },

            "latency": {

                "total": total_latency,

                "query_rewriting": (
                    rewrite_trace.get(
                        "rewriter_latency",
                        0.0,
                    )
                ),

                "retrieval": retrieval_latency,

                "generation": generation_latency,
            },
        }

        logger.info(
            "Question terminée | "
            "total_latency=%.3fs",
            total_latency,
        )

        logger.info(
            "=" * 70
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

            "query_type": (
                "rewritten"
                if rewrite_trace.get(
                    "rewritten",
                    False,
                )
                else "standalone"
            ),

            "retrieved_count": len(
                retrieval_results
            ),

            "relevant_count": len(
                relevant_documents
            ),

            "trace": trace,
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

            "relevance_threshold": (
                self.relevance_threshold
            ),

            "history_enabled": (
                self.use_history
            ),

            "history_messages": len(
                self.history
            ),

            "query_rewriting": (
                self.enable_query_rewriting
            ),

            "retriever_initialized": (
                self.retriever is not None
            ),

            "generator_initialized": (
                self.generator is not None
            ),
        }

