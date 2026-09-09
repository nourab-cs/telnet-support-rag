from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_ollama import ChatOllama


class QueryRewriter:
    """
    Reformule uniquement les questions qui dépendent du contexte précédent.

    La décision de savoir si la question dépend de l'historique est faite
    dans RAGPipeline. Cette classe est donc appelée uniquement lorsque
    la question a besoin d'être reformulée.
    """

    def __init__(
        self,
        model_name: str = "mistral",
        temperature: float = 0.0,
        max_history_chars: int = 4000,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.max_history_chars = max_history_chars

        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature,
        )

    def rewrite(
        self,
        question: str,
        history: Optional[str] = None,
    ) -> str:
        """
        Transforme une question dépendante de l'historique en requête
        autonome pour la recherche.

        La méthode ne doit jamais répondre à la question.
        """
        original_question = (question or "").strip()

        if not original_question:
            return original_question

        history_text = (history or "").strip()

        if not history_text:
            return original_question

        # Limite la taille de l'historique utilisé par le rewriter.
        history_text = history_text[-self.max_history_chars:]

        prompt = f"""
Tu es un module de reformulation de requêtes pour un moteur RAG.

Historique de conversation :
{history_text}

Question actuelle :
{original_question}

Ta tâche :
- La question actuelle dépend déjà du contexte précédent.
- Transforme-la uniquement en une requête de recherche autonome.
- Conserve exactement l'intention de la question.
- Utilise les informations utiles présentes dans l'historique pour remplacer
  les pronoms ou références ambiguës.
- N'ajoute aucune information qui n'est pas présente dans l'historique.
- Ne réponds JAMAIS à la question.
- Ne donne aucune commande, procédure, explication ou solution.
- Retourne UNE SEULE phrase courte, adaptée à une recherche documentaire.
- Maximum 15 mots.
- N'utilise pas de guillemets ni de préfixe comme "Requête :".

Exemples :

Historique :
Utilisateur : Comment installer SmartConnect ?
Question : Comment vérifier son installation ?
Sortie :
Comment vérifier l'installation de SmartConnect ?

Historique :
Utilisateur : Le port utilisé par SmartConnect est 8443.
Question : Comment en vérifier le port ?
Sortie :
Comment vérifier le port 8443 de SmartConnect ?

Historique :
Utilisateur : Comment installer SmartConnect ?
Question : Lister les devices.
Sortie :
Lister les devices.

Historique :
Utilisateur : Comment installer SmartConnect ?
Question : Comment faire une recherche ?
Sortie :
Comment faire une recherche ?

Réponds uniquement avec la requête finale.
""".strip()

        try:
            response = self.llm.invoke(prompt)

            if hasattr(response, "content"):
                rewritten = response.content
            else:
                rewritten = str(response)

            rewritten = self._clean_output(rewritten)

            if not rewritten:
                return original_question

            return rewritten

        except Exception:
            # En cas d'erreur du LLM, on conserve la question originale.
            return original_question

    @staticmethod
    def _clean_output(text: str) -> str:
        """
        Nettoie la sortie du modèle sans modifier son sens.
        """
        text = (text or "").strip()

        if not text:
            return ""

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        if not lines:
            return ""

        text = lines[0]

        prefixes = (
            "Requête :",
            "Requete :",
            "Requête:",
            "Requete:",
            "Query:",
            "Query :",
            "Réponse :",
            "Reponse :",
        )

        for prefix in prefixes:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()
                break

        if (
            len(text) >= 2
            and text[0] == text[-1]
            and text[0] in {'"', "'"}
        ):
            text = text[1:-1].strip()

        text = " ".join(text.split())

        return text
