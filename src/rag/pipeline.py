from __future__ import annotations

import logging
import re
from typing import Any, List, Tuple

from langchain_core.documents import Document

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

        Question
            ↓
        Détection conversationnelle
            ↓
        Historique
            ↓
        Détection question indirecte
            ↓
        Query enrichie
            ↓
        Retrieval
            ↓
        Relevance Gate
            ↓
        Déduplication
            ↓
        Limitation du contexte
            ↓
        Generator
            ↓
        Réponse + sources
    """

    # ============================================================
    # CONFIGURATION
    # ============================================================

    CHUNKING_VERSION = "2.0"

    FALLBACK_RESPONSE = (
        "Je ne trouve pas cette information dans la documentation."
    )

    CONVERSATIONAL_RESPONSE = (
        "Bonjour ! Je suis l’assistant technique "
        "TELNET SmartConnect. Comment puis-je vous aider ?"
    )

    # ============================================================
    # INITIALISATION
    # ============================================================

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

        # --------------------------------------------------------
        # Composants
        # --------------------------------------------------------

        self.loader: DocumentLoader | None = None
        self.embedder: Embedder | None = None
        self.chunker: ChunkingAgent | None = None
        self.store: ChromaStore | None = None

        self.vectordb = None
        self.retriever: Retriever | None = None
        self.generator: Generator | None = None
        self.history: ConversationHistory | None = None

        # --------------------------------------------------------
        # Retrieval
        # --------------------------------------------------------

        self.retrieval_k = 8
        self.retrieval_fetch_k = 20

        # 1.0 = pertinence
        # 0.0 = diversité
        self.retrieval_lambda = 0.6

        # Seuil normal
        self.relevance_threshold = 0.40

        # Pour une question indirecte :
        # seuil légèrement plus souple.
        self.indirect_relevance_threshold = 0.25

        # Nombre maximum de chunks envoyés au LLM
        self.max_context_documents = 6

        # --------------------------------------------------------
        # Document Loader
        # --------------------------------------------------------

        self.loader = DocumentLoader(
            data_dir=self.data_dir
        )

        # --------------------------------------------------------
        # Embeddings
        # --------------------------------------------------------

        self.embedder = Embedder(
            model_name="BAAI/bge-m3",
            device="cpu",
            normalize_embeddings=True,
            cache_folder="./models_cache",
        )

        embeddings = self.embedder.get_embeddings()

        # --------------------------------------------------------
        # Chunking
        # --------------------------------------------------------

        self.chunker = ChunkingAgent(
            model_name="mistral",
            embeddings=embeddings,
            enable_llm_analysis=False,
            target_chunk_size=900,
            min_chunk_size=300,
            max_chunk_size=1200,
            cache_dir="./chunking_cache"
        )

        # --------------------------------------------------------
        # Chroma
        # --------------------------------------------------------

        self.store = ChromaStore(
            persist_directory=self.db_dir,
            collection_name=self.collection_name,
        )

        # --------------------------------------------------------
        # Historique
        # --------------------------------------------------------

        if self.use_history:

            self.history = ConversationHistory(
                max_history=self.max_history
            )

    # ============================================================
    # CONVERSATION
    # ============================================================

    @staticmethod
    def is_conversational(
        question: str,
    ) -> bool:
        """
        Détecte les messages conversationnels simples.

        Ils ne nécessitent pas le RAG.
        """

        if not question:
            return False

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
            "ça va",
            "comment ça va",
        }

        return text in conversational_inputs

    @staticmethod
    def conversational_response(
        question: str,
    ) -> str:

        text = question.lower().strip()

        if text in {
            "bonjour",
            "bonsoir",
            "salut",
            "hello",
            "hi",
            "hey",
            "coucou",
        }:

            return (
                "Bonjour ! Je suis l’assistant technique "
                "TELNET SmartConnect. Comment puis-je vous aider ?"
            )

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

            return (
                "Au revoir ! N’hésitez pas à revenir "
                "si vous avez besoin d’aide."
            )

        if text in {
            "ça va",
            "comment ça va",
        }:

            return (
                "Je vais bien, merci ! "
                "Comment puis-je vous aider avec "
                "TELNET SmartConnect ?"
            )

        return (
            "Bonjour ! Je suis l’assistant technique "
            "TELNET SmartConnect. Comment puis-je vous aider ?"
        )

    # ============================================================
    # QUESTION INDIRECTE
    # ============================================================

    @staticmethod
    def is_indirect_question(
        question: str,
    ) -> bool:
        """
        Détecte les questions qui dépendent probablement
        du contexte précédent.

        Exemples :

            Et après ?
            Et ensuite ?
            Après l'installation ?
            Et pour la configuration ?
            Comment faire ensuite ?
            Et cette erreur ?
            Et si ça ne fonctionne pas ?
            C'est quoi ça ?
        """

        if not question:
            return False

        text = question.lower().strip()

        # --------------------------------------------------------
        # Marqueurs très fréquents
        # --------------------------------------------------------

        # Simple check: if it starts with "et ", it's likely indirect
        if text.startswith("et "):
            return True

        indirect_patterns = [
            r"^et (après|apres)",  # "et apres", "et apres installation"
            r"^et ensuite\b",
            r"^et maintenant\b",
            r"^(après|apres)",  # "apres", "apres installation"
            r"^ensuite\b",
            r"^puis\b",
            r"^et pour\b",
            r"^et concernant\b",
            r"^et qu'en est[- ]il\b",
            r"^qu'en est[- ]il\b",
            r"^comment faire ensuite\b",
            r"^que faire ensuite\b",
            r"^que faire (après|apres)\b",
            r"^et si\b",
            r"^et cette\b",
            r"^et ce\b",
            r"^et ça\b",
            r"^et cela\b",
            r"^c'est quoi ça\b",
            r"^c'est quoi cela\b",
            r"^ça\b",
            r"^cela\b",
            r"^et .+",  # General pattern: "et" followed by anything
            r"(après|apres)\s+",  # Alternative pattern without anchor
            r"et\s+(après|apres)",  # More flexible pattern
            r"et\s+\w+",  # "et" followed by any word (catches "et apres installation")
        ]

        for pattern in indirect_patterns:

            if re.search(
                pattern,
                text,
            ):
                return True

        # --------------------------------------------------------
        # Questions très courtes
        # --------------------------------------------------------

        words = text.split()

        if len(words) <= 5:

            short_markers = {
                "après",
                "apres",
                "ensuite",
                "puis",
                "maintenant",
                "ça",
                "cela",
                "ceci",
                "cette",
                "cette étape",
                "et après",
                "et apres",
                "et ensuite",
                "et maintenant",
            }

            if text in short_markers:
                return True

        return False

    # ============================================================
    # BUILD INDEX
    # ============================================================

    def build_index(self):

        if self.loader is None:
            raise RuntimeError(
                "DocumentLoader non initialisé."
            )

        if self.embedder is None:
            raise RuntimeError(
                "Embedder non initialisé."
            )

        if self.chunker is None:
            raise RuntimeError(
                "ChunkingAgent non initialisé."
            )

        if self.store is None:
            raise RuntimeError(
                "ChromaStore non initialisé."
            )

        # --------------------------------------------------------
        # 1. CHARGEMENT
        # --------------------------------------------------------

        logger.info(
            "Chargement des documents..."
        )

        documents = self.loader.load()

        if not documents:
            raise ValueError(
                "Aucun document trouvé dans le dossier data."
            )

        logger.info(
            "Documents chargés : %d",
            len(documents),
        )

        # --------------------------------------------------------
        # 2. CHUNKING
        # --------------------------------------------------------

        logger.info(
            "Découpage des documents..."
        )

        chunks = self.chunker.chunk_documents(
            documents
        )

        if not chunks:
            raise ValueError(
                "Le chunking n'a produit aucun chunk."
            )

        logger.info(
            "Chunks générés : %d",
            len(chunks),
        )

        # --------------------------------------------------------
        # 3. EMBEDDINGS
        # --------------------------------------------------------

        embeddings = self.embedder.get_embeddings()

        # --------------------------------------------------------
        # 4. CHROMA
        # --------------------------------------------------------

        logger.info(
            "Création de la base vectorielle..."
        )

        self.vectordb = self.store.create_or_replace(
            chunks=chunks,
            embeddings=embeddings,
            documents=documents,
            embedding_model=self.embedder.model_name,
            chunking_version=self.CHUNKING_VERSION,
        )

        # --------------------------------------------------------
        # 5. QUERY COMPONENTS
        # --------------------------------------------------------

        self._initialize_query_components()

        logger.info(
            "Index RAG construit avec succès."
        )

        return self.vectordb

    # ============================================================
    # LOAD EXISTING
    # ============================================================

    def load_existing(self):

        if self.store is None:
            raise RuntimeError(
                "ChromaStore non initialisé."
            )

        if self.embedder is None:
            raise RuntimeError(
                "Embedder non initialisé."
            )

        if self.loader is None:
            raise RuntimeError(
                "DocumentLoader non initialisé."
            )

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

        logger.info(
            "Index existant chargé."
        )

        return self.vectordb

    # ============================================================
    # INITIALISATION QUERY
    # ============================================================

    def _initialize_query_components(self):

        if self.vectordb is None:
            raise RuntimeError(
                "La base vectorielle n'est pas disponible."
            )

        self.retriever = Retriever(
            vectordb=self.vectordb,
            k=self.retrieval_k,
            search_type="mmr",
            score_threshold=self.relevance_threshold,
            fetch_k=self.retrieval_fetch_k,
            lambda_mult=self.retrieval_lambda,
        )

        self.generator = Generator(
            retriever=self.retriever.retriever,
            model_name="mistral",
            temperature=0.0,
            num_predict=512,
        )

        logger.info(
            "Retriever et Generator initialisés."
        )

    # ============================================================
    # DEDUPLICATION
    # ============================================================

    @staticmethod
    def _deduplicate_documents(
        documents_with_scores: List[
            Tuple[Document, float]
        ],
    ) -> List[
        Tuple[Document, float]
    ]:

        unique_results = []
        seen = set()

        for document, score in documents_with_scores:

            content = (
                document.page_content.strip()
            )

            if not content:
                continue

            # Hash simple du contenu
            content_key = content

            if content_key in seen:
                continue

            seen.add(content_key)

            unique_results.append(
                (document, score)
            )

        return unique_results

    # ============================================================
    # CONSTRUCTION QUERY
    # ============================================================

    def _build_search_query(
        self,
        question: str,
    ) -> Tuple[str, bool]:
        """
        Construit la requête utilisée par le Retriever.

        Retourne :

            search_query
            is_indirect

        Exemple :

            Question précédente :
            "Comment installer SmartConnect ?"

            Question actuelle :
            "Et après l'installation ?"

            Requête finale :

            "Comment installer SmartConnect ?
             Et après l'installation ?"
        """

        # First check if the question is indirect regardless of history
        is_indirect = self.is_indirect_question(question)

        # If not using history or no history available, return as is
        if not self.use_history or not self.history:
            return question, is_indirect

        recent_questions = (
            self.history.get_recent_questions(
                n=2
            )
        )

        if not recent_questions:
            return question, is_indirect

        # --------------------------------------------------------
        # QUESTION DIRECTE
        # --------------------------------------------------------

        if not is_indirect:

            logger.info(
                "Question directe détectée."
            )

            return question, False

        # --------------------------------------------------------
        # QUESTION INDIRECTE
        # --------------------------------------------------------

        # On utilise seulement les dernières questions
        # pour éviter de polluer la requête.
        context_questions = recent_questions[-2:]

        context_str = " ".join(
            context_questions
        )

        search_query = (
            f"{context_str} {question}"
        )

        logger.info(
            "Question indirecte détectée."
        )

        logger.info(
            "Contexte utilisé pour le retrieval : %s",
            context_str,
        )

        logger.info(
            "Requête enrichie : %s",
            search_query,
        )

        return search_query, True

    # ============================================================
    # ASK
    # ============================================================

    def ask(
        self,
        question: str,
    ) -> dict[str, Any]:

        # ========================================================
        # 1. VALIDATION
        # ========================================================

        if not question or not question.strip():

            return {
                "answer": "Veuillez entrer une question.",
                "source_documents": [],
                "sources_used": 0,
                "retrieval_scores": [],
                "query_type": "invalid",
                "documents_retrieved": 0,
                "documents_relevant": 0,
            }

        question = question.strip()

        logger.info(
            "Question reçue : %s",
            question,
        )

        # ========================================================
        # 2. CONVERSATION
        # ========================================================

        if self.is_conversational(question):

            answer = self.conversational_response(
                question
            )

            if self.use_history and self.history:

                self.history.add_user_message(
                    question
                )

                self.history.add_assistant_message(
                    answer,
                    sources=[]  # Conversational responses don't have sources
                )

            return {
                "answer": answer,
                "source_documents": [],
                "sources_used": 0,
                "retrieval_scores": [],
                "query_type": "conversation",
                "documents_retrieved": 0,
                "documents_relevant": 0,
            }

        # ========================================================
        # 3. VÉRIFICATION
        # ========================================================

        if self.retriever is None:

            raise RuntimeError(
                "Le retriever n'est pas initialisé. "
                "Construisez ou chargez l'index."
            )

        if self.generator is None:

            raise RuntimeError(
                "Le generator n'est pas initialisé."
            )

        # ========================================================
        # 4. HISTORIQUE POUR LE GENERATOR
        # ========================================================

        history_text = ""

        if self.use_history and self.history:

            history_text = (
                self.history.get_context_for_prompt(
                    max_chars=2000
                )
            )

        # ========================================================
        # 5. CONSTRUCTION QUERY
        # ========================================================

        search_query, is_indirect = (
            self._build_search_query(
                question
            )
        )

        # ========================================================
        # 6. RETRIEVAL
        # ========================================================

        logger.info(
            "Recherche documentaire..."
        )

        retrieval_results = (
            self.retriever.retrieve_with_scores(
                query=search_query,
                k=self.retrieval_k,
            )
        )

        documents_retrieved = len(
            retrieval_results
        )

        logger.info(
            "Documents récupérés : %d",
            documents_retrieved,
        )

        # ========================================================
        # 7. RELEVANCE GATE
        # ========================================================

        relevant_results = []

        # Pour une question indirecte, on utilise un seuil
        # légèrement plus permissif.
        if is_indirect:

            current_threshold = (
                self.indirect_relevance_threshold
            )

            logger.info(
                "Question indirecte : "
                "seuil de pertinence ajusté à %.2f",
                current_threshold,
            )

        else:

            current_threshold = (
                self.relevance_threshold
            )

        # Si le score le plus élevé est proche du seuil mais inférieur,
        # on ajuste le seuil pour inclure au moins un document
        if retrieval_results and current_threshold > 0.20:
            max_score = max(score for _, score in retrieval_results)
            if max_score >= current_threshold - 0.10 and max_score < current_threshold:
                # Ajuster le seuil pour inclure le meilleur document
                current_threshold = max_score
                logger.info(
                    "Ajustement du seuil de pertinence à %.2f "
                    "pour inclure le meilleur document (score: %.2f)",
                    current_threshold,
                    max_score,
                )

        for document, score in retrieval_results:

            source = document.metadata.get(
                "source",
                document.metadata.get(
                    "file_name",
                    "unknown",
                ),
            )

            logger.info(
                "Score %.4f | source=%s",
                score,
                source,
            )

            if score >= current_threshold:

                relevant_results.append(
                    (document, score)
                )

        logger.info(
            "Relevance Gate : %d/%d documents conservés "
            "avec threshold=%.2f.",
            len(relevant_results),
            documents_retrieved,
            current_threshold,
        )

        # ========================================================
        # 8. FALLBACK
        # ========================================================

        if not relevant_results:

            logger.info(
                "Aucun document suffisamment pertinent."
            )

            answer = self.FALLBACK_RESPONSE

            if self.use_history and self.history:

                self.history.add_user_message(
                    question
                )

                self.history.add_assistant_message(
                    answer,
                    sources=[]  # Fallback responses don't have sources
                )

            return {
                "answer": answer,
                "source_documents": [],
                "sources_used": 0,
                "retrieval_scores": [],
                "query_type": (
                    "indirect"
                    if is_indirect
                    else "technical"
                ),
                "documents_retrieved": documents_retrieved,
                "documents_relevant": 0,
            }

        # ========================================================
        # 9. DEDUPLICATION
        # ========================================================

        unique_results = (
            self._deduplicate_documents(
                relevant_results
            )
        )

        logger.info(
            "Après déduplication : %d documents.",
            len(unique_results),
        )

        # ========================================================
        # 10. LIMITATION CONTEXTE
        # ========================================================

        final_results = unique_results[
            :self.max_context_documents
        ]

        relevant_documents = [
            document
            for document, score in final_results
        ]

        final_scores = [
            float(score)
            for document, score in final_results
        ]

        logger.info(
            "Documents envoyés au Generator : %d",
            len(relevant_documents),
        )

        # ========================================================
        # 11. GENERATION
        # ========================================================

        logger.info(
            "Génération de la réponse..."
        )

        result = self.generator.generate(
            question=question,
            documents=relevant_documents,
            history=history_text,
        )

        if isinstance(result, dict):

            answer = result.get(
                "answer",
                self.FALLBACK_RESPONSE,
            )

        else:

            answer = str(result)

        if not answer or not answer.strip():

            answer = self.FALLBACK_RESPONSE

        # ========================================================
        # 12. HISTORIQUE
        # ========================================================

        if self.use_history and self.history:

            self.history.add_user_message(
                question
            )

            # Extract sources from relevant documents
            sources = [
                doc.metadata.get("source", "inconnu")
                for doc in relevant_documents
            ]

            self.history.add_assistant_message(
                answer,
                sources=sources
            )

        # ========================================================
        # 13. RESULTAT FINAL
        # ========================================================

        return {
            "answer": answer,
            "source_documents": relevant_documents,
            "sources_used": len(
                relevant_documents
            ),
            "retrieval_scores": final_scores,
            "query_type": (
                "indirect"
                if is_indirect
                else "technical"
            ),
            "documents_retrieved": documents_retrieved,
            "documents_relevant": len(
                relevant_documents
            ),
            "search_query": search_query,
        }

    # ============================================================
    # HISTORY
    # ============================================================

    def get_history(self) -> str:
        """
        Retourne l'historique formaté.
        """

        if not self.history:
            return ""

        return self.history.format_history()

    def get_history_summary(self) -> str:
        """
        Retourne un résumé de l'historique.
        """

        if not self.history:
            return "Aucun historique disponible."

        return self.history.get_summary()

    def clear_history(self):

        if self.history:

            self.history.clear()

            logger.info(
                "Historique effacé."
            )

    # ============================================================
    # STATUS
    # ============================================================

    def get_status(self) -> dict[str, Any]:

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
            "retrieval_fetch_k": self.retrieval_fetch_k,
            "retrieval_lambda": self.retrieval_lambda,
            "relevance_threshold": (
                self.relevance_threshold
            ),
            "indirect_relevance_threshold": (
                self.indirect_relevance_threshold
            ),
            "max_context_documents": (
                self.max_context_documents
            ),
        }