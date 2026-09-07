from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from typing import Optional, List
import re
import logging

logger = logging.getLogger(__name__)


class Generator:
    """
    Gère le LLM et la chaîne RAG avec l'API moderne LCEL et prompts améliorés.
    """

    FALLBACK_MESSAGE = "Je ne trouve pas cette information dans la documentation."

    def __init__(
        self,
        retriever,
        model_name: str = "mistral",
        temperature: float = 0.0,
        num_predict: int = 512,
    ):

        self.retriever = retriever
        self.use_history = False
        self.chain = None
        self.model_name = model_name

        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature,
            num_predict=num_predict,
        )

    def create_chain(self, use_history: bool = False):

        # Template amélioré et équilibré
        if use_history:
            template = """Tu es l'assistant technique du support TELNET SmartConnect.

CONTEXTE DOCUMENTAIRE:
{context}

Historique de conversation:
{history}

QUESTION:
{question}

Réponds en français en te basant sur le contexte documentaire. Si l'information n'est pas dans le contexte, indique-le clairement. Sois précis et utile."""
        else:
            template = """Tu es l'assistant technique du support TELNET SmartConnect.

CONTEXTE DOCUMENTAIRE:
{context}

QUESTION:
{question}

Réponds en français en te basant sur le contexte documentaire. Si l'information n'est pas dans le contexte, indique-le clairement. Sois précis et utile."""

        prompt = ChatPromptTemplate.from_template(template)

        # Créer la chaîne avec l'API LCEL moderne
        if use_history:
            self.chain = (
                {
                    "context": self.retriever | self.format_docs_improved,
                    "question": RunnablePassthrough(),
                    "history": lambda x: x.get("history", "")
                }
                | prompt
                | self.llm
                | StrOutputParser()
            )
        else:
            self.chain = (
                {
                    "context": self.retriever | self.format_docs_improved,
                    "question": RunnablePassthrough()
                }
                | prompt
                | self.llm
                | StrOutputParser()
            )

        self.use_history = use_history
        return self

    def format_docs(self, docs):
        """Formatage simple (ancienne méthode)."""
        return "\n\n".join(doc.page_content for doc in docs)

    def format_docs_improved(self, docs: List[Document]) -> str:
        """Formatage amélioré avec métadonnées et scores."""
        if not docs:
            return ""

        sections = []
        for index, doc in enumerate(docs, start=1):
            metadata = doc.metadata
            filename = metadata.get("filename") or metadata.get("source", "source inconnue")
            score = metadata.get("retrieval_score")

            header = f"[Document {index} - {filename}"
            if score is not None:
                header += f" (pertinence: {score:.2f})"
            header += "]"

            section = f"{header}\n{doc.page_content.strip()}"
            sections.append(section)

        return "\n\n---\n\n".join(sections)

    def _contains_suspicious_content(self, text: str) -> bool:
        """Détecte du contenu suspect (maths, hors sujet, etc.)."""
        suspicious_patterns = [
            r'Let\s+\w+\s*=',
            r'Suppose\s+',
            r'Question:\s*Let',
            r'What\s+is\s+the\s+\d+\s*\+\s*\d+',  # Only catch obvious math problems
        ]

        for pattern in suspicious_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                logger.warning(f"Contenu suspect détecté avec pattern: {pattern}")
                return True

        return False
    
    def _validate_response_length(self, text: str, max_length: int = 1000) -> bool:
        """Valide la longueur de la réponse."""
        if len(text) > max_length:
            logger.warning(f"Réponse trop longue: {len(text)} > {max_length} caractères")
            return False
        return True
    
    def _validate_response_quality(self, text: str) -> bool:
        """Valide la qualité de la réponse."""
        # Vérifier que la réponse n'est pas vide
        if not text or not text.strip():
            logger.warning("Réponse vide")
            return False
        
        # Vérifier que ce n'est pas le message d'erreur standard répété
        if text.count("Je ne trouve pas cette information") > 1:
            logger.warning("Message d'erreur répété")
            return False
        
        # Vérifier qu'il n'y a pas de répétitions excessives
        words = text.split()
        if len(words) > 10:
            unique_words = set(words)
            if len(unique_words) / len(words) < 0.3:  # Moins de 30% de mots uniques
                logger.warning("Trop de répétitions dans la réponse")
                return False
        
        return True

    def invoke(self, question: str, history: Optional[str] = None):
        """Invoke the chain and return both result and sources"""
        # Recreate chain only if history status changed
        has_history = history and history.strip()
        if self.chain is None or (has_history != self.use_history):
            self.create_chain(use_history=has_history)

        # Get sources first - use invoke instead of get_relevant_documents
        sources = self.retriever.invoke(question)

        logger.info(f"{len(sources)} documents récupérés pour la question: {question}")
        
        # Debug: print retrieved context
        if sources:
            context_length = sum(len(doc.page_content) for doc in sources)
            logger.info(f"Context length: {context_length} chars")
            logger.info(f"Number of documents: {len(sources)}")
            logger.debug(f"Context preview: {self.format_docs_improved(sources[:2])[:500]}")

        # Get the answer with history if provided
        if has_history:
            result = self.chain.invoke({"question": question, "history": history})
        else:
            result = self.chain.invoke(question)

        # Valider la réponse
        result = result.strip()

        # Validation allégée - seulement validation critique
        validation_passed = True

        # 1. Validation de contenu suspect (très stricte)
        if self._validate_response_quality(result) and self._contains_suspicious_content(result):
            logger.warning("Validation contenu suspect échouée")
            validation_passed = False

        # Utiliser le fallback seulement si validation échouée
        if not validation_passed:
            logger.warning("Utilisation du message fallback suite à validation échouée")
            result = self.FALLBACK_MESSAGE

        return {
            "result": result,
            "source_documents": sources,
            "validation_passed": validation_passed
        }

    def generate(self, question: str, documents: List[Document], history: Optional[str] = None):
        """Generate response using pre-retrieved documents (for new pipeline architecture)"""
        # Format the documents for the prompt
        context = self.format_docs_improved(documents)
        
        # Template amélioré et équilibré
        if history and history.strip():
            template = """Tu es l'assistant technique du support TELNET SmartConnect.

CONTEXTE DOCUMENTAIRE:
{context}

Historique de conversation:
{history}

QUESTION:
{question}

Réponds en français en te basant sur le contexte documentaire. Si l'information n'est pas dans le contexte, indique-le clairement. Sois précis et utile."""
        else:
            template = """Tu es l'assistant technique du support TELNET SmartConnect.

CONTEXTE DOCUMENTAIRE:
{context}

QUESTION:
{question}

Réponds en français en te basant sur le contexte documentaire. Si l'information n'est pas dans le contexte, indique-le clairement. Sois précis et utile."""

        prompt = ChatPromptTemplate.from_template(template)
        
        # Always pass a dictionary to the chain
        input_data = {
            "context": context,
            "question": question,
            "history": history if history and history.strip() else ""
        }
        
        # Create chain for this specific context
        chain = (
            {
                "context": lambda x: x["context"],
                "question": lambda x: x["question"],
                "history": lambda x: x["history"]
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        result = chain.invoke(input_data)

        # Valider la réponse
        result = result.strip()

        # Validation allégée - seulement validation critique
        validation_passed = True

        # 1. Validation de contenu suspect (très stricte)
        if self._validate_response_quality(result) and self._contains_suspicious_content(result):
            logger.warning("Validation contenu suspect échouée")
            validation_passed = False

        # Utiliser le fallback seulement si validation échouée
        if not validation_passed:
            logger.warning("Utilisation du message fallback suite à validation échouée")
            result = self.FALLBACK_MESSAGE

        return {
            "answer": result,
            "source_documents": documents,
            "validation_passed": validation_passed
        }