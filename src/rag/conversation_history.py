from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional
import json


class ConversationHistory:
    """
    Gestion de l'historique d'une session de conversation.
    """

    def __init__(
        self,
        max_history: int = 10,
    ):

        if max_history <= 0:
            raise ValueError(
                "max_history doit être > 0."
            )

        self.max_history = max_history

        self.messages: List[
            Dict[str, str]
        ] = []

        self.session_id = (
            datetime.now()
            .strftime("%Y%m%d_%H%M%S")
        )

    def add_message(
        self,
        role: str,
        content: str,
        sources: Optional[List[str]] = None,
    ):

        if role not in {
            "user",
            "assistant",
        }:
            raise ValueError(
                "role doit être 'user' ou 'assistant'."
            )

        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }

        if sources and role == "assistant":
            message["sources"] = sources

        self.messages.append(message)

        max_messages = (
            self.max_history * 2
        )

        if len(self.messages) > max_messages:
            self.messages = (
                self.messages[-max_messages:]
            )

    def add_user_message(
        self,
        question: str,
    ):

        self.add_message(
            "user",
            question,
        )

    def add_assistant_message(
        self,
        answer: str,
        sources: Optional[List[str]] = None,
    ):

        self.add_message(
            "assistant",
            answer,
            sources,
        )

    def get_history(
        self,
        last_n: Optional[int] = None,
    ):

        if last_n is None:
            return self.messages.copy()

        return self.messages[-last_n:]

    def get_formatted_history(
        self,
        last_n: Optional[int] = None,
    ) -> str:

        messages = self.get_history(
            last_n
        )

        formatted = []

        for message in messages:

            if message["role"] == "user":

                formatted.append(
                    f"Utilisateur : "
                    f"{message['content']}"
                )

            else:

                formatted.append(
                    f"Assistant : "
                    f"{message['content']}"
                )

        return "\n".join(formatted)

    def get_context_for_prompt(
        self,
        max_chars: int = 6000,
    ) -> str:

        formatted = (
            self.get_formatted_history()
        )

        if len(formatted) <= max_chars:
            return formatted

        return formatted[-max_chars:]

    def get_last_question(
        self,
    ) -> Optional[str]:

        for message in reversed(
            self.messages
        ):

            if message["role"] == "user":
                return message["content"]

        return None

    def get_last_answer(
        self,
    ) -> Optional[str]:

        for message in reversed(
            self.messages
        ):

            if message["role"] == "assistant":
                return message["content"]

        return None

    def clear(self):

        self.messages = []

    def get_conversation_summary(
        self,
    ) -> str:

        if not self.messages:
            return (
                "Aucune conversation en cours."
            )

        questions = [
            m["content"]
            for m in self.messages
            if m["role"] == "user"
        ]

        answers = [
            m["content"]
            for m in self.messages
            if m["role"] == "assistant"
        ]

        return (
            f"Session : {self.session_id}\n"
            f"Questions : {len(questions)}\n"
            f"Réponses : {len(answers)}\n"
            f"Dernière question : "
            f"{questions[-1] if questions else 'N/A'}"
        )

    def save_to_file(
        self,
        filepath: str,
    ):

        with open(
            filepath,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                {
                    "session_id": self.session_id,
                    "messages": self.messages,
                },
                file,
                ensure_ascii=False,
                indent=2,
            )

    def load_from_file(
        self,
        filepath: str,
    ):

        with open(
            filepath,
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

        self.session_id = data.get(
            "session_id",
            self.session_id,
        )

        self.messages = data.get(
            "messages",
            [],
        )

    def __len__(self):
        return len(self.messages)
