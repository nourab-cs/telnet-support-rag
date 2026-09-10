
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_ollama import ChatOllama

from .chunking_agent import ChunkingAgent
from .chroma_store import ChromaStore
from .conversation_history import ConversationHistory
from .document_loader import DocumentLoader
from .embedder import Embedder
from .generator import Generator
from .query_rewriter import QueryRewriter
from .retriever import Retriever


logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Pipeline RAG TELNET SmartConnect.

    Architecture :

        Question
            ↓
        Validation
            ↓
        History Dependency
            ↓
        Query Rewriting
            ↓
        Retrieval
            ↓
        Relevance Gate
            ↓
        Deduplication
            ↓
        Context
            ↓
        Generator
            ↓
        Answer
            ↓
        History
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

        relevance_threshold: Optional[float] = 0.40,
        bm25_relevance_threshold: Optional[float] = None,
        hybrid_relevance_threshold: Optional[float] = None,
        similarity_score_threshold: Optional[float] = 0.40,

        max_context_documents: int = 6,

        use_history: bool = True,
        max_history: int = 10,
        max_history_chars: int = 5000,

        enable_query_rewriting: bool = True,
    ) -> None:

        # ========================================================
        # VALIDATION CONFIGURATION
        # ========================================================

        if retrieval_type not in Retriever.VALID_SEARCH_TYPES:
            raise ValueError(
                f"retrieval_type invalide : {retrieval_type}"
            )

        if int(retrieval_k) <= 0:
            raise ValueError(
                "retrieval_k doit être > 0."
            )

        if int(retrieval_fetch_k) <= 0:
            raise ValueError(
                "retrieval_fetch_k doit être > 0."
            )

        if not 0.0 <= float(retrieval_lambda) <= 1.0:
            raise ValueError(
                "retrieval_lambda doit être compris entre 0 et 1."
            )

        if int(hybrid_k) <= 0:
            raise ValueError(
                "hybrid_k doit être > 0."
            )

        if int(max_context_documents) <= 0:
            raise ValueError(
                "max_context_documents doit être > 0."
            )

        for name, value in {
            "relevance_threshold": relevance_threshold,
            "bm25_relevance_threshold": bm25_relevance_threshold,
            "hybrid_relevance_threshold": hybrid_relevance_threshold,
        }.items():

            if value is not None and float(value) < 0:
                raise ValueError(
                    f"{name} doit être >= 0 ou None."
                )

        if similarity_score_threshold is not None:
            if not 0.0 <= float(similarity_score_threshold) <= 1.0:
                raise ValueError(
                    "similarity_score_threshold doit être compris "
                    "entre 0 et 1."
                )

        # ========================================================
        # CONFIGURATION
        # ========================================================

        self.data_dir = data_dir
        self.db_dir = db_dir
        self.collection_name = collection_name

        self.embedding_model = embedding_model
        self.llm_model = llm_model

        self.retrieval_type = retrieval_type
        self.retrieval_k = int(retrieval_k)
        self.retrieval_fetch_k = int(retrieval_fetch_k)
        self.retrieval_lambda = float(retrieval_lambda)
        self.hybrid_k = int(hybrid_k)

        self.relevance_threshold = relevance_threshold
        self.bm25_relevance_threshold = bm25_relevance_threshold
        self.hybrid_relevance_threshold = hybrid_relevance_threshold
        self.similarity_score_threshold = similarity_score_threshold

        self.max_context_documents = int(
            max_context_documents
        )

        self.use_history = use_history
        self.max_history = int(max_history)
        self.max_history_chars = int(max_history_chars)

        self.enable_query_rewriting = (
            enable_query_rewriting
        )

        self.chunking_version = "chunking_agent_v1"

        self.index_documents: List[
            Document
        ] = []

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
            enable_llm_analysis=False,
            target_chunk_size=900,
            min_chunk_size=300,
            max_chunk_size=1200,
        )

        # ========================================================
        # CHROMA
        # ========================================================

        self.vector_store = ChromaStore(
            persist_directory=db_dir,
            collection_name=collection_name,
        )

        # ========================================================
        # COMPONENTS
        # ========================================================

        self.retriever: Optional[
            Retriever
        ] = None

        self.generator: Optional[
            Generator
        ] = None

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
        # QUESTION VALIDATOR
        # ========================================================

        self.question_validator = ChatOllama(
            model=llm_model,
            temperature=0.0,
            num_predict=160,
        )

    # ============================================================
    # JSON
    # ============================================================

    @staticmethod
    def _parse_llm_json(
        content: Any,
    ) -> Dict[str, Any]:

        text = str(
            content
        ).strip()

        if text.startswith("```"):

            lines = text.splitlines()

            if (
                lines
                and lines[0].strip().startswith("```")
            ):
                lines = lines[1:]

            if (
                lines
                and lines[-1].strip() == "```"
            ):
                lines = lines[:-1]

            text = "\n".join(
                lines
            ).strip()

        start = text.find("{")

        if start == -1:
            raise ValueError(
                "Aucun objet JSON trouvé dans la réponse du LLM."
            )

        decoder = json.JSONDecoder()

        data, _ = decoder.raw_decode(
            text[start:]
        )

        if not isinstance(
            data,
            dict,
        ):
            raise ValueError(
                "La réponse JSON du LLM n'est pas un objet."
            )

        return data

    @staticmethod
    def _parse_bool(
        value: Any,
        default: bool = False,
    ) -> bool:

        if isinstance(
            value,
            bool,
        ):
            return value

        if isinstance(
            value,
            str,
        ):

            value = value.strip().lower()

            if value == "true":
                return True

            if value == "false":
                return False

        return default

    # ============================================================
    # QUESTION VALIDATION
    # ============================================================

    def _validate_question(
        self,
        question: str,
        history: str = "",
    ) -> Dict[str, Any]:

        history_context = (
            history or ""
        ).strip()

        if len(history_context) > 3500:
            history_context = history_context[-3500:]

        prompt = f"""
Tu es un classificateur pour un assistant de support
TELNET SmartConnect.

Retourne UNIQUEMENT un JSON valide :

{{
  "type": "smartconnect_question" | "conversational" | "invalid",
  "reason": "courte explication"
}}

Règles :

1. conversational :
   salutations, remerciements, au revoir et petites formules sociales.

2. smartconnect_question :
   - toute demande liée au support SmartConnect/TELNET ;
   - questions techniques ou fonctionnelles plausiblement couvertes
     par la documentation ;
   - API, authentification, JWT, token, appareils, configuration,
     connexion, tableau de bord, erreurs, paramètres, etc. ;
   - une question de suivi doit être considérée comme technique
     si l'historique montre qu'elle continue une discussion SmartConnect.

3. invalid :
   - texte incompréhensible ;
   - bruit aléatoire ;
   - question clairement sans rapport avec SmartConnect/TELNET.

4. Ne rejette PAS une question simplement parce que
   "SmartConnect" ou "TELNET" n'est pas écrit.

HISTORIQUE :
{history_context if history_context else "(aucun historique)"}

QUESTION :
{question}

JSON :
"""

        try:

            response = (
                self.question_validator.invoke(
                    prompt
                )
            )

            content = getattr(
                response,
                "content",
                response,
            )

            data = self._parse_llm_json(
                content
            )

            question_type = str(
                data.get(
                    "type",
                    "invalid",
                )
            ).strip().lower()

            reason = str(
                data.get(
                    "reason",
                    "",
                )
            ).strip()

            if question_type not in {
                "smartconnect_question",
                "conversational",
                "invalid",
            }:
                question_type = "invalid"

            return {
                "type": question_type,
                "reason": reason,
                "validator_ok": True,
            }

        except Exception as exc:

            logger.warning(
                "Question Validator error | question=%r | error=%s",
                question,
                exc,
            )

            return {
                "type": "smartconnect_question",
                "reason": "validator_error_fallback",
                "validator_ok": False,
                "error": str(exc),
            }

    # ============================================================
    # CONVERSATIONAL
    # ============================================================

    def _conversational_response(
        self,
        question: str,
    ) -> str:

        prompt = f"""
Tu es un assistant de support TELNET SmartConnect.

Réponds brièvement et naturellement à cette interaction.
Ne parle pas de documents, de RAG ou de recherche.

Message :
{question}

Réponse :
"""

        try:

            response = (
                self.question_validator.invoke(
                    prompt
                )
            )

            content = getattr(
                response,
                "content",
                response,
            )

            return str(
                content
            ).strip()

        except Exception:

            return (
                "Bonjour ! Comment puis-je vous aider "
                "concernant SmartConnect ?"
            )

    # ============================================================
    # HISTORY DEPENDENCY
    # ============================================================

    def _question_depends_on_history(
        self,
        question: str,
        history: str,
    ) -> Dict[str, Any]:

        if (
            not self.use_history
            or not history.strip()
        ):
            return {
                "needs_history": False,
                "reason": "no_history",
            }

        if self.query_rewriter is None:
            return {
                "needs_history": False,
                "reason": "rewriter_unavailable",
            }

        prompt = f"""
Analyse si la question suivante dépend de l'historique.

Retourne UNIQUEMENT un JSON valide :

{{
  "needs_history": true ou false,
  "reason": "courte explication"
}}

Réponds true si la question utilise :
- un pronom ;
- une référence implicite ;
- un élément déjà discuté ;
- une formulation clairement liée à la question précédente.

Réponds false si la question est autonome.

HISTORIQUE :
{history}

QUESTION :
{question}

JSON :
"""

        try:

            response = (
                self.query_rewriter.llm.invoke(
                    prompt
                )
            )

            content = getattr(
                response,
                "content",
                response,
            )

            data = self._parse_llm_json(
                content
            )

            return {
                "needs_history": self._parse_bool(
                    data.get(
                        "needs_history",
                        False,
                    )
                ),
                "reason": str(
                    data.get(
                        "reason",
                        "",
                    )
                ).strip(),
                "classifier_ok": True,
            }

        except Exception as exc:

            logger.warning(
                "History Dependency error | question=%r | error=%s",
                question,
                exc,
            )

            return {
                "needs_history": False,
                "reason": "classifier_error_fallback",
                "classifier_ok": False,
                "error": str(exc),
            }

    # ============================================================
    # INITIALISATION
    # ============================================================

    def _initialize_query_components(
        self,
    ) -> None:

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

        retriever_documents = (
            self.index_documents
            if self.retrieval_type
            in ("bm25", "hybrid")
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
            similarity_score_threshold=(
                self.similarity_score_threshold
            ),
        )

        self.generator = Generator(
            model_name=self.llm_model,
            temperature=0.0,
            num_predict=512,
        )

    # ============================================================
    # BUILD INDEX
    # ============================================================

    def build_index(
        self,
    ) -> Dict[str, Any]:

        start = time.perf_counter()

        documents = self.loader.load()

        if not documents:
            raise ValueError(
                "Aucun document trouvé."
            )

        chunks = (
            self.chunker.chunk_documents(
                documents
            )
        )

        if not chunks:
            raise ValueError(
                "Aucun chunk généré."
            )

        self.vector_store.create_or_replace(
            chunks=chunks,
            embeddings=self.embeddings,
            documents=documents,
            embedding_model=self.embedding_model,
            chunking_version=self.chunking_version,
        )

        self.index_documents = list(
            chunks
        )

        self._initialize_query_components()

        elapsed = (
            time.perf_counter()
            - start
        )

        logger.info(
            "Index construit | documents=%d | chunks=%d | latency=%.3fs",
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

    def load_existing(
        self,
        strict_compatibility: bool = True,
    ) -> None:

        if strict_compatibility:

            compatible = (
                self.vector_store.is_compatible(
                    embedding_model=self.embedding_model,
                    chunking_version=self.chunking_version,
                )
            )

            if not compatible:

                manifest = (
                    self.vector_store.get_manifest()
                )

                if manifest:
                    raise ValueError(
                        "Index Chroma incompatible avec la configuration actuelle : "
                        f"embedding={manifest.get('embedding_model')!r}, "
                        f"chunking={manifest.get('chunking_version')!r}."
                    )

                raise ValueError(
                    "Manifest Chroma absent. "
                    "Utilisez strict_compatibility=False "
                    "ou reconstruisez l'index."
                )

        self.vector_store.load(
            embeddings=self.embeddings
        )

        self.index_documents = (
            self.vector_store.get_documents()
        )

        if not self.index_documents:
            raise ValueError(
                "Aucun chunk récupérable depuis l'index Chroma."
            )

        self._initialize_query_components()

        logger.info(
            "Index existant chargé | chunks=%d",
            len(self.index_documents),
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

            logger.warning(
                "Query rewriting error | question=%r | error=%s",
                original_question,
                exc,
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

        return (
            rewritten_query,
            {
                "enabled": True,
                "rewritten": was_rewritten,
                "original_query": original_question,
                "search_query": rewritten_query,
                "history_used": True,
                "rewriter_latency": elapsed,
            },
        )

    # ============================================================
    # SCORE TYPE
    # ============================================================

    def _get_score_type(self) -> str:

        if self.retrieval_type == "bm25":
            return "bm25"

        if self.retrieval_type == "hybrid":
            return "rrf"

        return "vector_relevance"

    # ============================================================
    # RELEVANCE THRESHOLD
    # ============================================================

    def _get_relevance_threshold(
        self,
    ) -> Optional[float]:

        if self.retrieval_type == "bm25":
            return self.bm25_relevance_threshold

        if self.retrieval_type == "hybrid":
            return self.hybrid_relevance_threshold

        if self.retrieval_type == "similarity_score_threshold":
            return self.similarity_score_threshold

        return self.relevance_threshold

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
            self._get_relevance_threshold()
        )

        score_type = (
            self._get_score_type()
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
                    "score_type": score_type,
                },
            )

        relevant = []
        rejected = []

        for document, score in results:

            score = float(
                score
            )

            if (
                threshold is None
                or score >= threshold
            ):

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
            "score_type": score_type,
        }

        logger.info(
            "Relevance Gate | "
            "type=%s | input=%d | accepted=%d | "
            "rejected=%d | threshold=%s",
            score_type,
            len(results),
            len(relevant),
            len(rejected),
            threshold,
        )

        return (
            relevant,
            metadata,
        )

    def _score_is_accepted(
        self,
        score: float,
    ) -> bool:

        threshold = (
            self._get_relevance_threshold()
        )

        return (
            threshold is None
            or float(score) >= threshold
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
                or metadata.get("relative_path")
                or metadata.get("file_path")
                or metadata.get("filename")
                or ""
            )

            chunk_id = (
                metadata.get("chunk_id")
                or ""
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
    # ASK
    # ============================================================

    def ask(
        self,
        question: str,
    ) -> Dict[str, Any]:

        start_total = time.perf_counter()

        question = (
            question or ""
        ).strip()

        if not question:
            raise ValueError(
                "La question ne peut pas être vide."
            )

        logger.info("=" * 70)

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
        # VALIDATION
        # ========================================================

        validation_trace = (
            self._validate_question(
                question=question,
                history=history_text,
            )
        )

        question_type = validation_trace.get(
            "type",
            "invalid",
        )

        # ========================================================
        # CONVERSATIONAL
        # ========================================================

        if question_type == "conversational":

            answer = (
                self._conversational_response(
                    question
                )
            )

            if self.use_history:

                self.history.add_user_message(
                    question
                )

                self.history.add_assistant_message(
                    answer,
                    sources=[],
                )

            total_latency = (
                time.perf_counter()
                - start_total
            )

            return {
                "answer": answer,
                "question": question,
                "search_query": question,
                "documents": [],
                "scores": [],
                "source_documents": [],
                "retrieval_scores": [],
                "query_type": "conversational",
                "retrieved_count": 0,
                "relevant_count": 0,
                "trace": {
                    "original_question": question,
                    "validation": validation_trace,
                    "history_dependency": {
                        "needs_history": False,
                        "reason": "conversational",
                    },
                    "query_rewriting": {
                        "enabled": self.enable_query_rewriting,
                        "rewritten": False,
                        "original_query": question,
                        "search_query": question,
                        "reason": "not_needed_conversational",
                    },
                    "retrieval": {
                        "configured_method": self.retrieval_type,
                        "k": self.retrieval_k,
                        "result_count": 0,
                        "executed": False,
                        "reason": "conversational",
                    },
                    "relevance_gate": {
                        "input_count": 0,
                        "accepted_count": 0,
                        "rejected_count": 0,
                        "applied": False,
                        "reason": "conversational",
                    },
                    "documents": [],
                    "context": {
                        "selected_count": 0,
                        "scores": [],
                    },
                    "generation": {
                        "executed": True,
                        "reason": "conversational",
                    },
                    "history": {
                        "enabled": self.use_history,
                        "messages": len(self.history),
                    },
                    "latency": {
                        "total": total_latency,
                        "query_rewriting": 0.0,
                        "retrieval": 0.0,
                        "generation": total_latency,
                    },
                },
            }

        # ========================================================
        # INVALID
        # ========================================================

        if question_type == "invalid":

            answer = (
                "Je n'ai pas compris votre demande. "
                "Veuillez reformuler votre question concernant "
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

            total_latency = (
                time.perf_counter()
                - start_total
            )

            return {
                "answer": answer,
                "question": question,
                "search_query": question,
                "documents": [],
                "scores": [],
                "source_documents": [],
                "retrieval_scores": [],
                "query_type": "invalid",
                "retrieved_count": 0,
                "relevant_count": 0,
                "trace": {
                    "original_question": question,
                    "validation": validation_trace,
                    "history_dependency": {
                        "needs_history": False,
                        "reason": "invalid",
                    },
                    "query_rewriting": {
                        "enabled": self.enable_query_rewriting,
                        "rewritten": False,
                        "original_query": question,
                        "search_query": question,
                        "reason": "not_needed_invalid",
                    },
                    "retrieval": {
                        "configured_method": self.retrieval_type,
                        "k": self.retrieval_k,
                        "result_count": 0,
                        "executed": False,
                        "reason": "invalid",
                    },
                    "relevance_gate": {
                        "input_count": 0,
                        "accepted_count": 0,
                        "rejected_count": 0,
                        "applied": False,
                        "reason": "invalid",
                    },
                    "documents": [],
                    "context": {
                        "selected_count": 0,
                        "scores": [],
                    },
                    "generation": {
                        "executed": False,
                        "reason": "invalid",
                    },
                    "history": {
                        "enabled": self.use_history,
                        "messages": len(self.history),
                    },
                    "latency": {
                        "total": total_latency,
                        "query_rewriting": 0.0,
                        "retrieval": 0.0,
                        "generation": 0.0,
                    },
                },
            }

        # ========================================================
        # HISTORY DEPENDENCY
        # ========================================================

        history_dependency_trace = (
            self._question_depends_on_history(
                question=question,
                history=history_text,
            )
        )

        needs_history = bool(
            history_dependency_trace.get(
                "needs_history",
                False,
            )
        )

        # ========================================================
        # QUERY REWRITING
        # ========================================================

        if needs_history:

            (
                search_query,
                rewrite_trace,
            ) = self._rewrite_query(
                question
            )

        else:

            search_query = question

            rewrite_trace = {
                "enabled": self.enable_query_rewriting,
                "rewritten": False,
                "original_query": question,
                "search_query": question,
                "history_used": False,
                "reason": "question_independent_of_history",
            }

            # Une question autonome ne doit pas être polluée
            # par l'historique dans le Generator.
            history_text = ""

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

        score_type = (
            self._get_score_type()
        )

        for rank, (
            document,
            score,
        ) in enumerate(
            retrieval_results,
            start=1,
        ):

            logger.info(
                "RESULT %d | score=%.6f | type=%s | source=%s | preview=%s",
                rank,
                float(score),
                score_type,
                (document.metadata or {}).get(
                    "source"
                ),
                (document.page_content or "")[
                    :100
                ].replace(
                    "\n",
                    " ",
                ),
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

        documents_with_scores: List[
            Tuple[Document, float]
        ] = []

        for document, score in relevant_results:

            content = (
                document.page_content or ""
            ).strip()

            metadata = (
                document.metadata or {}
            )

            source = (
                metadata.get("source")
                or metadata.get("relative_path")
                or metadata.get("file_path")
                or metadata.get("filename")
                or ""
            )

            chunk_id = (
                metadata.get(
                    "chunk_id"
                )
                or ""
            )

            key = (
                source,
                chunk_id,
                content,
            )

            if key in seen:
                continue

            seen.add(key)

            document.metadata[
                "retrieval_score"
            ] = float(score)

            document.metadata[
                "retrieval_score_type"
            ] = score_type

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

        # ========================================================
        # TRACE DOCUMENTS
        # ========================================================

        documents_trace = []

        threshold = (
            self._get_relevance_threshold()
        )

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
                    "score_type": score_type,
                    "source": metadata.get(
                        "source"
                    ),
                    "file": metadata.get(
                        "filename"
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
                        threshold is None
                        or float(score) >= threshold
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
        # LATENCY
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

            "validation": validation_trace,

            "history_dependency": (
                history_dependency_trace
            ),

            "query_rewriting": rewrite_trace,

            "retrieval": {
                "configured_method": self.retrieval_type,
                "score_type": score_type,
                "k": self.retrieval_k,
                "fetch_k": self.retrieval_fetch_k,
                "lambda": self.retrieval_lambda,
                "hybrid_k": self.hybrid_k,
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
                "score_type": score_type,
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
                "query_rewriting": rewrite_trace.get(
                    "rewriter_latency",
                    0.0,
                ),
                "retrieval": retrieval_latency,
                "generation": generation_latency,
            },
        }

        logger.info(
            "Question terminée | total_latency=%.3fs",
            total_latency,
        )

        logger.info("=" * 70)

        # ========================================================
        # RESULT
        # ========================================================

        return {
            "answer": answer,
            "question": question,
            "search_query": search_query,

            "documents": relevant_documents,
            "scores": relevant_scores,

            "source_documents": relevant_documents,
            "retrieval_scores": relevant_scores,

            "retrieval_score_type": score_type,

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

    def clear_history(
        self,
    ) -> None:

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

    def status(
        self,
    ) -> Dict[str, Any]:

        return {
            "data_dir": self.data_dir,
            "db_dir": self.db_dir,
            "collection_name": self.collection_name,

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

            "hybrid_k": self.hybrid_k,

            "relevance_threshold": (
                self.relevance_threshold
            ),

            "bm25_relevance_threshold": (
                self.bm25_relevance_threshold
            ),

            "hybrid_relevance_threshold": (
                self.hybrid_relevance_threshold
            ),

            "similarity_score_threshold": (
                self.similarity_score_threshold
            ),

            "chunking_version": (
                self.chunking_version
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

