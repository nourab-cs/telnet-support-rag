from __future__ import annotations

import logging
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


logger = logging.getLogger(__name__)


class Generator:
    """
    Générateur de réponses avec Mistral via Ollama.

    Responsabilités :
    - construire le prompt
    - intégrer le contexte documentaire
    - intégrer l'historique
    - appeler le LLM
    - valider la réponse
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
    ):

        self.model_name = model_name

        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature,
            num_predict=num_predict,
        )

        self.chain = None
        self.use_history = False

    # ============================================================
    # CHAIN
    # ============================================================

    def create_chain(
        self,
        use_history: bool = False,
    ):

        if use_history:

            template = """
Tu es un assistant technique spécialisé
dans TELNET SmartConnect.

Ta mission est de répondre à la question
en utilisant UNIQUEMENT les informations
présentes dans la documentation fournie.

RÈGLES STRICTES :

1. Ne jamais inventer une information.
2. Ne jamais utiliser de connaissance externe.
3. Les mots similaires ne suffisent pas.
4. Le contexte doit réellement répondre à la question.
5. Ne fais pas de déduction non présente dans la documentation.
6. Si l'information n'est pas disponible, réponds exactement :

"Je ne trouve pas cette information dans la documentation."

7. Réponds en français.
8. Pour une procédure, utilise des étapes numérotées.
9. Conserve les commandes et paramètres techniques.
10. Ne mentionne jamais les chunks.
11. Ne mentionne jamais le contexte documentaire.

HISTORIQUE :

{history}

DOCUMENTATION :

{context}

QUESTION :

{question}

RÉPONSE :
"""

        else:

            template = """
Tu es un assistant technique spécialisé
dans TELNET SmartConnect.

Ta mission est de répondre à la question
en utilisant UNIQUEMENT les informations
présentes dans la documentation fournie.

RÈGLES STRICTES :

1. Ne jamais inventer une information.
2. Ne jamais utiliser de connaissance externe.
3. Les mots similaires ne suffisent pas.
4. Le contexte doit réellement répondre à la question.
5. Ne fais pas de déduction non présente dans la documentation.
6. Si l'information n'est pas disponible, réponds exactement :

"Je ne trouve pas cette information dans la documentation."

7. Réponds en français.
8. Pour une procédure, utilise des étapes numérotées.
9. Conserve les commandes et paramètres techniques.
10. Ne mentionne jamais les chunks.
11. Ne mentionne jamais le contexte documentaire.

DOCUMENTATION :

{context}

QUESTION :

{question}

RÉPONSE :
"""

        prompt = ChatPromptTemplate.from_template(
            template
        )

        if use_history:

            self.chain = (
                {
                    "context": lambda x: x["context"],
                    "question": lambda x: x["question"],
                    "history": lambda x: x["history"],
                }
                | prompt
                | self.llm
                | StrOutputParser()
            )

        else:

            self.chain = (
                {
                    "context": lambda x: x["context"],
                    "question": lambda x: x["question"],
                }
                | prompt
                | self.llm
                | StrOutputParser()
            )

        self.use_history = use_history

        return self

    # ============================================================
    # FORMATAGE
    # ============================================================

    def format_docs(
        self,
        docs: List[Document],
    ) -> str:

        if not docs:
            return ""

        sections = []

        for index, doc in enumerate(
            docs,
            start=1,
        ):

            metadata = doc.metadata or {}

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

                header += (
                    f" | pertinence : {score:.2f}"
                )

            header += "]"

            sections.append(
                f"{header}\n"
                f"{doc.page_content.strip()}"
            )

        return "\n\n---\n\n".join(
            sections
        )

    # ============================================================
    # VALIDATION
    # ============================================================

    def _validate_response_quality(
        self,
        text: str,
    ) -> bool:

        if not text or not text.strip():
            return False

        if len(text.split()) > 10:

            words = text.lower().split()

            unique_ratio = (
                len(set(words))
                / len(words)
            )

            if unique_ratio < 0.30:
                return False

        return True

    # ============================================================
    # GENERATION
    # ============================================================

    def generate(
        self,
        question: str,
        documents: List[Document],
        history: Optional[str] = None,
    ) -> dict:

        if not question or not question.strip():
            raise ValueError(
                "La question ne peut pas être vide."
            )

        if not documents:

            return {
                "answer": self.FALLBACK_MESSAGE,
                "source_documents": [],
            }

        question = question.strip()

        context = self.format_docs(
            documents
        )

        use_history = bool(
            history and history.strip()
        )

        if (
            self.chain is None
            or self.use_history != use_history
        ):

            self.create_chain(
                use_history=use_history
            )

        inputs = {
            "question": question,
            "context": context,
        }

        if use_history:
            inputs["history"] = history

        result = self.chain.invoke(
            inputs
        )

        result = result.strip()

        if not self._validate_response_quality(
            result
        ):

            logger.warning(
                "Validation de réponse échouée."
            )

            result = self.FALLBACK_MESSAGE

        return {
            "answer": result,
            "source_documents": documents,
        }
