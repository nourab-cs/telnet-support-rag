
"""
TELNET Support Bot - RAG Pipeline

Pipeline principal :

Question
   ↓
Détection conversationnelle
   ↓
Retriever MMR
   ↓
Filtrage par pertinence
   ↓
Suppression des doublons
   ↓
Generator
   ↓
Réponse + sources
"""

from __future__ import annotations

import logging
from typing import Any

from .document_loader import DocumentLoader
from .chunking_agent import ChunkingAgent
from .embedder import Embedder
from .chroma_store import ChromaStore
from .retriever import Retriever
from .generator import Generator
from .conversation_history import ConversationHistory


logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Pipeline RAG complet pour TELNET SmartConnect.

    Architecture :

        Documents
            ↓
        DocumentLoader
            ↓
        ChunkingAgent
            ↓
        BGE-M3
            ↓
        Chroma
            ↓
        MMR Retriever
            ↓
        Relevance Gate
            ↓
        Generator
    """

    # Version du chunking utilisée pour l'index
    CHUNKING_VERSION = "2.0"

    # Réponse utilisée lorsqu'aucune information pertinente
    # n'est trouvée dans la documentation.
    FALLBACK_RESPONSE = (
        "Je ne trouve pas cette information dans la documentation."
    )

    # Réponse pour les conversations simples
    CONVERSATIONAL_RESPONSE = (
        "Bonjour ! Je suis l’assistant technique "
        "TELNET SmartConnect. Comment puis-je vous aider ?"
    )

    def __init__(
        self,
        data_dir: str,
        db_dir: str,
        collection_name: str = "telnet_support",
        use_history: bool = True,
        max_history: int = 10,
    ):
        self.data_dir = data_dir
        self.db_dir = db_dir
        self.collection_name = collection_name

        self.use_history = use_history
        self.max_history = max_history

        # --------------------------------------------------
        # Composants
        # --------------------------------------------------

        self.loader: DocumentLoader | None = None
        self.embedder: Embedder | None = None
        self.chunker: ChunkingAgent | None = None
        self.store: ChromaStore | None = None

        self.vectordb = None
        self.retriever: Retriever | None = None
        self.generator: Generator | None = None

        self.history: ConversationHistory | None = None

        # --------------------------------------------------
        # Paramètres
        # --------------------------------------------------

        self.retrieval_k = 8
        self.retrieval_fetch_k = 20
        self.retrieval_lambda = 0.6

        # IMPORTANT 
        # Cette valeur doit être calibrée sur ton dataset.
        self.relevance_threshold = 0.40

        # --------------------------------------------------
        # Initialisation des composants de base
        # --------------------------------------------------

        self.loader = DocumentLoader(
            data_dir=self.data_dir
        )

        self.embedder = Embedder(
            model_name="BAAI/bge-m3",
            device="cpu",
            normalize_embeddings=True,
            cache_folder="./models_cache",
        )

        # Le même modèle BGE-M3 est partagé avec le ChunkingAgent.
        embeddings = self.embedder.get_embeddings()

        self.chunker = ChunkingAgent(
            embeddings=embeddings
        )

        self.store = ChromaStore(
            persist_directory=self.db_dir,
            collection_name=self.collection_name,
        )

        if self.use_history:
            self.history = ConversationHistory(
                max_history=self.max_history
            )

    # ======================================================
    # CONVERSATION
    # ======================================================

    @staticmethod
    def is_conversational(question: str) -> bool:
        """
        Détecte les messages conversationnels simples.

        Ces questions ne nécessitent pas de recherche
        dans la base documentaire.
        """

        text = question.lower().strip()

        conversational_inputs = {
            "bonjour",
            "bonsoir",
            "salut",
            "hello",
            "hi",
            "hey",
            "coucou",
            "merci",
            "merci beaucoup",
            "thanks",
            "thank you",
            "au revoir",
            "bye",
            "ok",
            "okay",
        }

        return text in conversational_inputs

    @staticmethod
    def conversational_response(question: str) -> str:
        """
        Retourne une réponse adaptée à une interaction simple.
        """

        text = question.lower().strip()

        if text in {
            "merci",
            "merci beaucoup",
            "thanks",
            "thank you",
        }:
            return "Avec plaisir !"

        if text in {
            "au revoir",
            "bye",
        }:
            return "Au revoir ! N’hésitez pas à revenir si vous avez besoin d’aide."

        return (
            "Bonjour ! Je suis l’assistant technique "
            "TELNET SmartConnect. Comment puis-je vous aider ?"
        )

    # ======================================================
    # BUILD INDEX
    # ======================================================

    def build_index(self):
        """
        Construit complètement l'index Chroma.

        Étapes :

        Documents
          ↓
        Loader
          ↓
        Chunking
          ↓
        Embeddings
          ↓
        Chroma
        """

        if self.loader is None:
            raise RuntimeError("DocumentLoader non initialisé.")

        if self.embedder is None:
            raise RuntimeError("Embedder non initialisé.")

        if self.chunker is None:
            raise RuntimeError("ChunkingAgent non initialisé.")

        if self.store is None:
            raise RuntimeError("ChromaStore non initialisé.")

        logger.info("Chargement des documents...")

        documents = self.loader.load()

        if not documents:
            raise ValueError(
                "Aucun document trouvé dans le dossier data."
            )

        logger.info(
            "Documents chargés : %d",
            len(documents)
        )

        # --------------------------------------------------
        # Chunking
        # --------------------------------------------------

        logger.info("Découpage des documents...")

        chunks = self.chunker.chunk_documents(
            documents
        )

        if not chunks:
            raise ValueError(
                "Le chunking n'a produit aucun chunk."
            )

        logger.info(
            "Chunks générés : %d",
            len(chunks)
        )

        # --------------------------------------------------
        # Chroma
        # --------------------------------------------------

        embeddings = self.embedder.get_embeddings()

        logger.info("Création de la base vectorielle...")

        self.vectordb = self.store.create_or_replace(
            chunks=chunks,
            embeddings=embeddings,
            documents=documents,
            embedding_model=self.embedder.model_name,
            chunking_version=self.CHUNKING_VERSION,
        )

        # --------------------------------------------------
        # Composants de recherche
        # --------------------------------------------------

        self._initialize_query_components()

        logger.info("Index RAG construit avec succès.")

        return self.vectordb

    # ======================================================
    # LOAD EXISTING
    # ======================================================

    def load_existing(self):
        """
        Charge un index Chroma existant.

        Vérifie également que l'index correspond
        aux documents actuels.
        """

        if self.store is None:
            raise RuntimeError("ChromaStore non initialisé.")

        if self.embedder is None:
            raise RuntimeError("Embedder non initialisé.")

        if self.loader is None:
            raise RuntimeError("DocumentLoader non initialisé.")

        if not self.store.exists():
            raise FileNotFoundError(
                "Aucun index Chroma existant."
            )

        logger.info(
            "Vérification de la compatibilité de l'index..."
        )

        documents = self.loader.load()

        compatible = self.store.is_compatible(
            documents=documents,
            embedding_model=self.embedder.model_name,
            chunking_version=self.CHUNKING_VERSION,
        )

        if not compatible:
            raise ValueError(
                "L'index existant est incompatible "
                "avec les documents ou la configuration actuelle."
            )

        embeddings = self.embedder.get_embeddings()

        self.vectordb = self.store.load(
            embeddings=embeddings
        )

        self._initialize_query_components()

        logger.info("Index existant chargé.")

        return self.vectordb

    # ======================================================
    # INITIALIZE QUERY COMPONENTS
    # ======================================================

    def _initialize_query_components(self):
        """
        Initialise les composants utilisés pendant les requêtes.
        """

        if self.vectordb is None:
            raise RuntimeError(
                "La base vectorielle n'est pas disponible."
            )

        # --------------------------------------------------
        # Retriever MMR
        # --------------------------------------------------

        self.retriever = Retriever(
            vectordb=self.vectordb,
            k=self.retrieval_k,
            search_type="mmr",
            score_threshold=self.relevance_threshold,
            fetch_k=self.retrieval_fetch_k,
            lambda_mult=self.retrieval_lambda,
        )

        # --------------------------------------------------
        # Generator
        # --------------------------------------------------

        self.generator = Generator(
            model_name="mistral",
            temperature=0.0,
            num_predict=512,
        )

    # ======================================================
    # DEDUPLICATION
    # ======================================================

    @staticmethod
    def _deduplicate_documents(documents):
        """
        Supprime les chunks identiques.

        On utilise le contenu comme clé principale.
        """

        unique_documents = []
        seen = set()

        for document in documents:

            content = document.page_content.strip()

            if not content:
                continue

            content_key = content

            if content_key in seen:
                continue

            seen.add(content_key)
            unique_documents.append(document)

        return unique_documents

    # ======================================================
    # ASK
    # ======================================================

    def ask(self, question: str) -> dict[str, Any]:
        """
        Pose une question au système RAG.

        Pipeline :

        Question
            ↓
        Conversation ?
            ↓ non
        MMR
            ↓
        Relevance Gate
            ↓
        Generator
        """
       

        if not question or not question.strip():
            return {
                "answer": "Veuillez entrer une question.",
                "source_documents": [],
                "sources_used": 0,
                "retrieval_scores": [],
            }

        question = question.strip()

        # ==================================================
        # 1. CONVERSATIONAL QUERY
        # ==================================================

        if self.is_conversational(question):
            
            answer = self.conversational_response(
                question
            )

            if self.use_history and self.history:

                self.history.add_user_message(question)

                self.history.add_assistant_message(
                    answer
                )

            return {
                "answer": answer,
                "source_documents": [],
                "sources_used": 0,
                "retrieval_scores": [],
                "query_type": "conversation",
            }

        # ==================================================
        # 2. VÉRIFICATION DU RETRIEVER
        # ==================================================

        if self.retriever is None:
            raise RuntimeError(
                "Le retriever n'est pas initialisé."
            )

        if self.generator is None:
            raise RuntimeError(
                "Le generator n'est pas initialisé."
            )

        # ==================================================
        # 3. HISTORIQUE
        # ==================================================

        history_text = ""

        if self.use_history and self.history:

            history_text = (
                self.history.get_context_for_prompt(
                    max_chars=2000
                )
            )

        # ==================================================
        # 4. REFORMULATION AVEC HISTORIQUE
        # ==================================================

        # Si historique disponible, reformuler la question avec contexte
        search_query = question
        if history_text:
            # Créer une requête enrichie avec le contexte historique
            recent_questions = self.history.get_recent_questions(n=2)
            if recent_questions:
                context_str = " ".join(recent_questions)
                search_query = f"{context_str} {question}"
                logger.info(
                    "Requête enrichie avec historique : %s",
                    search_query
                )

        # ==================================================
        # 5. RETRIEVAL
        # ==================================================

        logger.info(
            "Recherche des documents pour : %s",
            question
        )

        retrieval_results = (
            self.retriever.retrieve_with_scores(
                question,
                k=self.retrieval_k,
            )
        )

        logger.info(
            "Documents récupérés : %d",
            len(retrieval_results)
        )

        # ==================================================
        # 6. RELEVANCE GATE
        # ==================================================

        relevant_results = []

        for document, score in retrieval_results:

            logger.debug(
                "Score %.4f | %s",
                score,
                document.metadata.get(
                    "source",
                    "unknown"
                ),
            )

            if score >= self.relevance_threshold:

                relevant_results.append(
                    (document, score)
                )

        # ==================================================
        # 7. AUCUN DOCUMENT PERTINENT
        # ==================================================

        if not relevant_results:

            logger.info(
                "Aucun document suffisamment pertinent."
            )

            answer = self.FALLBACK_RESPONSE

            if self.use_history and self.history:

                self.history.add_user_message(question)

                self.history.add_assistant_message(
                    answer
                )

            return {
                "answer": answer,
                "source_documents": [],
                "sources_used": 0,
                "retrieval_scores": [],
                "query_type": "technical",
                "documents_retrieved": len(
                    retrieval_results
                ),
                "documents_relevant": 0,
            }

        # ==================================================
        # 8. EXTRACTION DES DOCUMENTS
        # ==================================================

        relevant_documents = [
            document
            for document, score in relevant_results
        ]

        relevant_scores = [
            score
            for document, score in relevant_results
        ]

        # ==================================================
        # 9. DÉDUPLICATION
        # ==================================================

        relevant_documents = (
            self._deduplicate_documents(
                relevant_documents
            )
        )

        # Limiter le contexte envoyé au LLM
        relevant_documents = relevant_documents[:6]

        # Recalcul des scores correspondants
        final_scores = relevant_scores[
            :len(relevant_documents)
        ]

        logger.info(
            "Documents pertinents : %d",
            len(relevant_documents)
        )

        # ==================================================
        # 10. GENERATION
        # ==================================================

        result = self.generator.generate(
            question=question,
            documents=relevant_documents,
            history=history_text,
        )

        # Selon l'implémentation de Generator,
        # result peut être une string ou un dictionnaire.
        if isinstance(result, dict):

            answer = result.get(
                "answer",
                self.FALLBACK_RESPONSE
            )

        else:

            answer = str(result)

        # ==================================================
        # 10. HISTORIQUE
        # ==================================================

        if self.use_history and self.history:

            self.history.add_user_message(
                question
            )

            self.history.add_assistant_message(
                answer
            )

        # ==================================================
        # 11. RESULTAT FINAL
        # ==================================================

        return {
            "answer": answer,
            "source_documents": relevant_documents,
            "sources_used": len(relevant_documents),
            "retrieval_scores": final_scores,
            "query_type": "technical",
            "documents_retrieved": len(
                retrieval_results
            ),
            "documents_relevant": len(
                relevant_documents
            ),
        }

    # ======================================================
    # CONVERSATIONAL
    # ======================================================

    def is_conversational(self, question: str) -> bool:
        """
        Détecte si la question est une conversation simple
        (salutations, etc.).
        """
        question_lower = question.lower().strip()
        conversational_patterns = [
            "bonjour",
            "salut",
            "hello",
            "hi",
            "merci",
            "thanks",
            "au revoir",
            "bye",
            "ça va",
            "comment ça va",
        ]
        return any(
            pattern in question_lower
            for pattern in conversational_patterns
        )

    def conversational_response(self, question: str) -> str:
        """
        Génère une réponse conversationnelle simple.
        """
        question_lower = question.lower().strip()

        if any(word in question_lower for word in ["bonjour", "salut", "hello", "hi"]):
            return "Bonjour ! Je suis l'assistant technique TELNET SmartConnect. Comment puis-je vous aider ?"
        elif any(word in question_lower for word in ["merci", "thanks"]):
            return "Je vous en prie ! N'hésitez pas si vous avez d'autres questions."
        elif any(word in question_lower for word in ["au revoir", "bye"]):
            return "Au revoir ! Bonne journée."
        elif "ça va" in question_lower or "comment ça va" in question_lower:
            return "Je suis un assistant technique, donc je vais bien ! Comment puis-je vous aider avec TELNET SmartConnect ?"
        else:
            return self.CONVERSATIONAL_RESPONSE

    # ======================================================
    # CONVERSATIONAL
    # ======================================================

    def is_conversational(self, question: str) -> bool:
        """
        Détecte si la question est une conversation simple
        (salutations, etc.).
        """
        question_lower = question.lower().strip()
        conversational_patterns = [
            "bonjour",
            "salut",
            "hello",
            "hi",
            "merci",
            "thanks",
            "au revoir",
            "bye",
            "ça va",
            "comment ça va",
        ]
        return any(
            pattern in question_lower
            for pattern in conversational_patterns
        )

    def conversational_response(self, question: str) -> str:
        """
        Génère une réponse conversationnelle simple.
        """
        question_lower = question.lower().strip()

        if any(word in question_lower for word in ["bonjour", "salut", "hello", "hi"]):
            return "Bonjour ! Je suis l'assistant technique TELNET SmartConnect. Comment puis-je vous aider ?"
        elif any(word in question_lower for word in ["merci", "thanks"]):
            return "Je vous en prie ! N'hésitez pas si vous avez d'autres questions."
        elif any(word in question_lower for word in ["au revoir", "bye"]):
            return "Au revoir ! Bonne journée."
        elif "ça va" in question_lower or "comment ça va" in question_lower:
            return "Je suis un assistant technique, donc je vais bien ! Comment puis-je vous aider avec TELNET SmartConnect ?"
        else:
            return self.CONVERSATIONAL_RESPONSE

    # ======================================================
    # HISTORY
    # ======================================================

    def get_history(self) -> str:
        """Retourne l'historique formaté."""

        if not self.history:
            return ""

        return self.history.format_history()

    def get_history_summary(self) -> str:
        """Retourne un résumé simple de l'historique."""

        if not self.history:
            return "Aucun historique disponible."

        return self.history.get_summary()

    def clear_history(self):
        """Efface l'historique."""

        if self.history:
            self.history.clear()

    # ======================================================
    # INFO
    # ======================================================

    def get_status(self) -> dict[str, Any]:
        """
        Retourne l'état actuel du pipeline.
        """

        return {
            "data_dir": self.data_dir,
            "db_dir": self.db_dir,
            "collection_name": self.collection_name,
            "index_loaded": self.vectordb is not None,
            "retriever_initialized": (
                self.retriever is not None
            ),
            "generator_initialized": (
                self.generator is not None
            ),
            "history_enabled": self.use_history,
            "retrieval_k": self.retrieval_k,
            "relevance_threshold": (
                self.relevance_threshold
            ),
        }

