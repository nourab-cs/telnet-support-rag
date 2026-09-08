from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


class ConversationHistory:
    """
    Gestion de l'historique d'une conversation.

    Responsabilités :
    - stocker les messages ;
    - conserver les rôles user/assistant ;
    - limiter la taille de l'historique ;
    - fournir un contexte au QueryRewriter ;
    - fournir un contexte au Generator ;
    - sauvegarder / charger une conversation.

    Cette classe ne contient aucune logique de compréhension
    des questions.
    """

    def __init__(
        self,
        max_history: int = 10,
        max_context_chars: int = 6000,
    ) -> None:

        self.max_history = max_history
        self.max_context_chars = max_context_chars

        self.messages: List[Dict[str, Any]] = []

        self.session_id = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )

        logger.info(
            "ConversationHistory initialisée | session=%s | "
            "max_history=%d | max_context_chars=%d",
            self.session_id,
            self.max_history,
            self.max_context_chars,
        )

    # ============================================================
    # AJOUT DES MESSAGES
    # ============================================================

    def add_message(
        self,
        role: str,
        content: str,
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> None:

        if role not in {"user", "assistant", "system"}:
            raise ValueError(
                f"Role invalide: {role}. "
                f"Utiliser user, assistant ou system."
            )

        content = (content or "").strip()

        if not content:
            return

        message: Dict[str, Any] = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }

        if sources:
            message["sources"] = sources

        self.messages.append(message)

        self._trim_history()

    def add_user_message(self, content: str) -> None:
        self.add_message("user", content)

    def add_assistant_message(
        self,
        content: str,
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.add_message(
            "assistant",
            content,
            sources=sources,
        )

    # ============================================================
    # LIMITATION DE L'HISTORIQUE
    # ============================================================

    def _trim_history(self) -> None:
        """
        Conserve au maximum max_history échanges.

        Un échange = question utilisateur + réponse assistant.
        """

        max_messages = self.max_history * 2

        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]

    # ============================================================
    # LECTURE
    # ============================================================

    def get_history(
        self,
        last_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:

        if last_n is None:
            return list(self.messages)

        return list(self.messages[-last_n:])

    def get_formatted_history(
        self,
        last_n: Optional[int] = None,
    ) -> str:

        messages = self.get_history(last_n)

        if not messages:
            return ""

        lines = []

        for message in messages:
            role = message["role"]

            if role == "user":
                label = "Utilisateur"
            elif role == "assistant":
                label = "Assistant"
            else:
                label = "Système"

            lines.append(
                f"{label}: {message['content']}"
            )

        return "\n".join(lines)

    def format_history(self) -> str:
        return self.get_formatted_history()

    # ============================================================
    # CONTEXTE POUR QUERY REWRITER
    # ============================================================

    def get_context_for_rewriting(
        self,
        max_chars: int = 4000,
        max_messages: int = 6,
    ) -> str:
        """
        Retourne uniquement les derniers messages utiles au
        QueryRewriter.

        On évite de transmettre une conversation énorme au LLM.
        """

        messages = self.messages[-max_messages:]

        if not messages:
            return ""

        lines = []

        for message in messages:
            role = message["role"]

            if role == "user":
                label = "Utilisateur"
            elif role == "assistant":
                label = "Assistant"
            else:
                label = "Système"

            lines.append(
                f"{label}: {message['content']}"
            )

        context = "\n".join(lines)

        if len(context) > max_chars:
            context = context[-max_chars:]

        return context

    # ============================================================
    # CONTEXTE POUR GENERATOR
    # ============================================================

    def get_context_for_prompt(
        self,
        max_chars: Optional[int] = None,
    ) -> str:

        if max_chars is None:
            max_chars = self.max_context_chars

        context = self.get_formatted_history()

        if len(context) <= max_chars:
            return context

        return context[-max_chars:]

    # ============================================================
    # QUESTIONS RECENTES
    # ============================================================

    def get_recent_questions(
        self,
        n: int = 2,
    ) -> List[str]:

        questions = [
            message["content"]
            for message in self.messages
            if message["role"] == "user"
        ]

        return questions[-n:]

    def get_last_question(self) -> Optional[str]:

        for message in reversed(self.messages):
            if message["role"] == "user":
                return message["content"]

        return None

    def get_last_answer(self) -> Optional[str]:

        for message in reversed(self.messages):
            if message["role"] == "assistant":
                return message["content"]

        return None

    # ============================================================
    # RESUME
    # ============================================================

    def get_context_summary(
        self,
        max_pairs: int = 3,
    ) -> str:

        messages = self.messages[-(max_pairs * 2):]

        if not messages:
            return ""

        lines = []

        for message in messages:

            role = (
                "Utilisateur"
                if message["role"] == "user"
                else "Assistant"
            )

            lines.append(
                f"{role}: {message['content']}"
            )

        return "\n".join(lines)

    def get_conversation_summary(
        self,
        max_pairs: int = 3,
    ) -> str:
        return self.get_context_summary(max_pairs)

    def get_summary(
        self,
        max_pairs: int = 3,
    ) -> str:
        return self.get_conversation_summary(max_pairs)

    # ============================================================
    # GESTION
    # ============================================================

    def clear(self) -> None:
        self.messages.clear()

        logger.info(
            "Historique effacé | session=%s",
            self.session_id,
        )

    # ============================================================
    # PERSISTENCE
    # ============================================================

    def save(self, filepath: str) -> None:

        path = Path(filepath)
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        data = {
            "session_id": self.session_id,
            "max_history": self.max_history,
            "max_context_chars": self.max_context_chars,
            "messages": self.messages,
        }

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2,
            )

        logger.info(
            "Historique sauvegardé | path=%s",
            filepath,
        )

    @classmethod
    def load(
        cls,
        filepath: str,
    ) -> "ConversationHistory":

        path = Path(filepath)

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

        history = cls(
            max_history=data.get(
                "max_history",
                10,
            ),
            max_context_chars=data.get(
                "max_context_chars",
                6000,
            ),
        )

        history.session_id = data.get(
            "session_id",
            history.session_id,
        )

        history.messages = data.get(
            "messages",
            [],
        )

        history._trim_history()

        return history

    # ============================================================
    # UTILITAIRES
    # ============================================================

    def __len__(self) -> int:
        return len(self.messages)

    def __repr__(self) -> str:
        return (
            f"ConversationHistory("
            f"session_id={self.session_id!r}, "
            f"messages={len(self.messages)})"
        )


class ConversationMemory:
    """
    Gestion de plusieurs sessions de conversation.
    """

    def __init__(
        self,
        max_sessions: int = 5,
        max_history: int = 10,
        max_context_chars: int = 6000,
    ) -> None:

        self.max_sessions = max_sessions
        self.max_history = max_history
        self.max_context_chars = max_context_chars

        self.sessions: Dict[
            str,
            ConversationHistory
        ] = {}

        self.current_session_id: Optional[str] = None

    def create_session(self) -> ConversationHistory:

        history = ConversationHistory(
            max_history=self.max_history,
            max_context_chars=self.max_context_chars,
        )

        self.sessions[history.session_id] = history
        self.current_session_id = history.session_id

        self._trim_sessions()

        return history

    def get_current_session(self) -> ConversationHistory:

        if self.current_session_id is None:
            return self.create_session()

        if self.current_session_id not in self.sessions:
            return self.create_session()

        return self.sessions[self.current_session_id]

    def switch_session(
        self,
        session_id: str,
    ) -> ConversationHistory:

        if session_id not in self.sessions:
            raise ValueError(
                f"Session inconnue: {session_id}"
            )

        self.current_session_id = session_id

        return self.sessions[session_id]

    def delete_session(
        self,
        session_id: str,
    ) -> None:

        if session_id in self.sessions:
            del self.sessions[session_id]

        if self.current_session_id == session_id:
            self.current_session_id = None

    def list_sessions(self) -> List[str]:
        return list(self.sessions.keys())

    def _trim_sessions(self) -> None:

        while len(self.sessions) > self.max_sessions:

            oldest_session = next(
                iter(self.sessions)
            )

            del self.sessions[oldest_session]