from __future__ import annotations

import logging
from typing import Optional

from langchain_ollama import ChatOllama


logger = logging.getLogger(__name__)


class QueryRewriter:
    """
    Reformule une question conversationnelle en requête autonome
    destinée au moteur de recherche du RAG.

    Exemple conceptuel :

        Historique :
            "Comment installer SmartConnect ?"
            "Il faut d'abord télécharger..."

        Question :
            "Et après ?"

        Requête produite :
            "Quelles sont les étapes à suivre après l'installation
             de SmartConnect ?"

    Le modèle ne répond PAS à la question.
    Il reformule uniquement la requête.

    Cette classe ne contient volontairement aucune règle métier
    ni liste d'exemples.
    """

    def __init__(
        self,
        model_name: str = "mistral",
        temperature: float = 0.0,
        max_history_chars: int = 5000,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.max_history_chars = max_history_chars

        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature,
        )

        logger.info(
            "QueryRewriter initialisé | model=%s | temperature=%.2f | "
            "max_history_chars=%d",
            model_name,
            temperature,
            max_history_chars,
        )

    def rewrite(
        self,
        question: str,
        history: Optional[str] = None,
    ) -> str:
        """
        Transforme la question en requête autonome.

        Si aucun historique n'est disponible, la question originale
        est conservée.
        """

        question = (question or "").strip()

        if not question:
            return ""

        if not history or not history.strip():
            logger.debug(
                "QueryRewriter | aucun historique | query=%r",
                question,
            )
            return question

        history = history[-self.max_history_chars :]

        prompt = f"""
Tu es un composant de reformulation d'un système RAG.

Ta seule tâche est de transformer la question actuelle de l'utilisateur
en une requête autonome, claire et explicite pouvant être envoyée
directement à un moteur de recherche documentaire.

Utilise l'historique uniquement pour comprendre le contexte
conversationnel nécessaire.

Tu dois notamment être capable de résoudre naturellement :
- les pronoms ;
- les références à un élément mentionné précédemment ;
- les questions elliptiques ;
- les formulations incomplètes ;
- les dépendances avec les messages précédents.

Règles importantes :

1. Ne réponds PAS à la question.
2. Retourne UNIQUEMENT la requête reformulée.
3. N'invente aucune information.
4. N'ajoute aucune information absente de la conversation.
5. Ne change pas l'intention de l'utilisateur.
6. Si la question est déjà autonome, conserve son intention
   et reformule-la au minimum si nécessaire.
7. La requête doit être adaptée à une recherche documentaire RAG.
8. Ne mentionne pas l'historique dans ta réponse.
9. Ne donne aucune explication.
10. Ne donne aucun préambule.

Historique de conversation :
{history}

Question actuelle :
{question}

Requête autonome :
"""

        try:
            response = self.llm.invoke(prompt)

            rewritten = self._extract_content(response)

            if not rewritten:
                logger.warning(
                    "QueryRewriter | réponse vide | fallback question originale"
                )
                return question

            rewritten = self._clean_output(rewritten)

            logger.info(
                "Query rewriting | original=%r | rewritten=%r",
                question,
                rewritten,
            )

            return rewritten

        except Exception as exc:
            logger.exception(
                "Erreur QueryRewriter | fallback sur question originale | %s",
                exc,
            )
            return question

    @staticmethod
    def _extract_content(response) -> str:
        """
        Extrait le texte retourné par ChatOllama.
        """

        content = getattr(response, "content", response)

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts = []

            for item in content:
                if isinstance(item, str):
                    parts.append(item)

                elif isinstance(item, dict):
                    text = item.get("text")

                    if text:
                        parts.append(str(text))

            return " ".join(parts).strip()

        return str(content).strip()

    @staticmethod
    def _clean_output(text: str) -> str:
        """
        Nettoyage léger de la sortie du modèle.
        """

        text = text.strip()

        prefixes = [
            "Requête autonome:",
            "Requete autonome:",
            "Query:",
            "Standalone query:",
        ]

        for prefix in prefixes:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()

        if len(text) >= 2:
            if (
                text.startswith('"')
                and text.endswith('"')
            ) or (
                text.startswith("'")
                and text.endswith("'")
            ):
                text = text[1:-1].strip()

        return text