
from __future__ import annotations

import re
from typing import List, Optional

from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


class Generator:
    """
    Générateur de réponses RAG avec Mistral/Ollama.

    Le Generator ne réalise PAS la recherche documentaire.

    Le Retriever est exécuté par le Pipeline, puis les documents
    pertinents sont transmis à :

        generate(
            question,
            documents,
            history
        )

    Architecture :

        Retriever
             ↓
        Documents
             ↓
        Generator
             ↓
        Mistral
             ↓
        Answer
    """

    FALLBACK_MESSAGE = (
        "Je ne trouve pas cette information "
        "dans la documentation."
    )

    def __init__(
        self,
        model_name: str = "mistral",
        temperature: float = 0.0,
        num_predict: int = 512,
        max_response_chars: int = 4000,
    ) -> None:

        self.model_name = model_name
        self.max_response_chars = max_response_chars
        if max_response_chars <= 0:
            raise ValueError("max_response_chars doit être > 0.")

        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature,
            num_predict=num_predict,
        )

    # ============================================================
    # FORMAT DOCUMENTS
    # ============================================================

    def format_docs(
        self,
        docs: List[Document],
    ) -> str:
        """
        Formatage simple des documents.
        """

        if not docs:
            return ""

        return "\n\n".join(
            (
                doc.page_content or ""
            ).strip()
            for doc in docs
        )

    def format_docs_improved(
        self,
        docs: List[Document],
    ) -> str:
        """
        Formatage amélioré avec métadonnées et scores.
        """

        if not docs:
            return ""

        sections = []

        for index, doc in enumerate(
            docs,
            start=1,
        ):

            metadata = (
                doc.metadata or {}
            )

            filename = (
                metadata.get("filename")
                or metadata.get("source")
                or "source inconnue"
            )

            score = metadata.get(
                "retrieval_score"
            )

            header = (
                f"[Document {index} - {filename}"
            )

            if score is not None:

                try:

                    header += (
                        f" | pertinence: "
                        f"{float(score):.2f}"
                    )

                except (TypeError, ValueError):
                    pass

            header += "]"

            content = (
                doc.page_content or ""
            ).strip()

            sections.append(
                f"{header}\n{content}"
            )

        return "\n\n---\n\n".join(
            sections
        )

    # ============================================================
    # VALIDATION
    # ============================================================

    def _contains_suspicious_content(
        self,
        text: str,
    ) -> bool:
        """
        Détecte certaines sorties manifestement hors sujet.
        """

        suspicious_patterns = [
            r"^\s*(?:system|developer)\s*:",
            r"^\s*ignore\s+(?:all|previous|prior)\s+instructions",
            r"<\/?(?:system|developer|tool)>"
        ]

        for pattern in suspicious_patterns:

            if re.search(
                pattern,
                text,
                re.IGNORECASE,
            ):

                return True

        return False

    def _validate_response_length(
        self,
        text: str,
        max_length: int = 1000,
    ) -> bool:
        """
        Vérifie que la réponse n'est pas excessivement longue.
        """

        if len(text) > max_length:

            return False

        return True

    def _validate_response_quality(
        self,
        text: str,
    ) -> bool:
        """
        Vérifications basiques de qualité.
        """

        if not text or not text.strip():

            return False

        if text.count(
            "Je ne trouve pas cette information"
        ) > 1:

            return False

        words = text.split()

        if len(words) > 10:

            unique_words = set(
                words
            )

            ratio = (
                len(unique_words)
                / len(words)
            )

            if ratio < 0.30:

                return False

        return True

    def _validate_response(
        self,
        result: str,
    ) -> tuple[str, bool]:
        """
        Validation centrale de la réponse.
        """

        result = (
            result or ""
        ).strip()

        quality_ok = (
            self._validate_response_quality(
                result
            )
        )

        suspicious = self._contains_suspicious_content(result)
        length_ok = self._validate_response_length(
            result, max_length=self.max_response_chars
        )

        validation_passed = quality_ok and not suspicious and length_ok

        if not validation_passed:

            result = self.FALLBACK_MESSAGE

        return (
            result,
            validation_passed,
        )

    # ============================================================
    # PROMPT
    # ============================================================

    def _build_prompt(
        self,
        has_history: bool,
    ) -> ChatPromptTemplate:
        """
        Construit le prompt RAG.
        """

        if has_history:

            template = """
Tu es l'assistant technique du support TELNET SmartConnect.

Ta tâche est de répondre à la question de l'utilisateur
en utilisant UNIQUEMENT les informations présentes dans
le contexte documentaire.

CONTEXTE DOCUMENTAIRE :
{context}

HISTORIQUE DE CONVERSATION :
{history}

QUESTION ACTUELLE :
{question}

RÈGLES :

1. Réponds en français.
2. Utilise prioritairement le contexte documentaire.
3. L'historique sert uniquement à comprendre le contexte
   conversationnel de la question.
4. N'invente aucune information.
5. Si la documentation ne permet pas de répondre,
   indique clairement que l'information n'est pas disponible.
6. Sois précis, direct et utile.
7. Ne mentionne pas le processus interne de recherche.
"""

        else:

            template = """
Tu es l'assistant technique du support TELNET SmartConnect.

Ta tâche est de répondre à la question de l'utilisateur
en utilisant UNIQUEMENT les informations présentes dans
le contexte documentaire.

CONTEXTE DOCUMENTAIRE :
{context}

QUESTION :
{question}

RÈGLES :

1. Réponds en français.
2. Utilise uniquement les informations du contexte.
3. N'invente aucune information.
4. Si la documentation ne permet pas de répondre,
   indique clairement que l'information n'est pas disponible.
5. Sois précis, direct et utile.
6. Ne mentionne pas le processus interne de recherche.
"""

        return ChatPromptTemplate.from_template(
            template
        )

    # ============================================================
    # GENERATE
    # ============================================================

    def generate(
        self,
        question: str,
        documents: List[Document],
        history: Optional[str] = None,
        standalone_question: Optional[str] = None,
    ) -> dict:
        """
        Génère une réponse à partir de documents
        déjà récupérés par le Retriever.

        Args:
            question:
                Question originale de l'utilisateur.

            documents:
                Documents sélectionnés par le Pipeline.

            history:
                Historique conversationnel formaté.
        """

        question = (
            question or ""
        ).strip()

        if not question:

            raise ValueError(
                "La question ne peut pas être vide."
            )

        # --------------------------------------------------------
        # CONTEXT
        # --------------------------------------------------------

        context = (
            self.format_docs_improved(
                documents
            )
        )

        has_history = bool(
            history
            and history.strip()
        )

        prompt = self._build_prompt(
            has_history=has_history
        )

        # --------------------------------------------------------
        # INPUT
        # --------------------------------------------------------

        input_data = {
            "context": context,
            "question": (standalone_question or question).strip(),
            "history": (
                history
                if has_history
                else ""
            ),
        }

        # --------------------------------------------------------
        # LCEL
        # --------------------------------------------------------

        chain = (
            prompt
            | self.llm
            | StrOutputParser()
        )

        # --------------------------------------------------------
        # INVOKE
        # --------------------------------------------------------

        try:

            result = chain.invoke(
                input_data
            )

        except Exception as exc:

            raise RuntimeError(
                "Erreur pendant la génération "
                "de la réponse."
            ) from exc

        # --------------------------------------------------------
        # VALIDATION
        # --------------------------------------------------------

        result, validation_passed = (
            self._validate_response(
                result
            )
        )

        return {
            "answer": result,
            "source_documents": documents,
            "validation_passed": validation_passed,
        }

