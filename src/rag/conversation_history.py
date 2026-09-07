"""
Gestion de l'historique de conversation pour le RAG conversationnel.

Ce module permet de maintenir le contexte des conversations précédentes
pour améliorer les réponses avec mémoire.
"""

from typing import List, Dict, Optional
from datetime import datetime
import json


class ConversationHistory:
    """
    Gère l'historique des conversations avec les messages utilisateur et assistant.
    """

    def __init__(self, max_history: int = 10):
        """
        Initialise l'historique de conversation.

        Args:
            max_history: Nombre maximum de paires question-réponse à conserver
        """
        self.max_history = max_history
        self.messages: List[Dict[str, str]] = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def add_message(self, role: str, content: str, sources: Optional[List[str]] = None):
        """
        Ajoute un message à l'historique.

        Args:
            role: "user" ou "assistant"
            content: Contenu du message
            sources: Sources utilisées pour la réponse (assistant uniquement)
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }

        if sources and role == "assistant":
            message["sources"] = sources

        self.messages.append(message)

        # Garder seulement les max_history derniers messages
        if len(self.messages) > self.max_history * 2:  # *2 car on compte les paires
            self.messages = self.messages[-self.max_history * 2:]

    def add_user_message(self, question: str):
        """Ajoute une question utilisateur."""
        self.add_message("user", question)

    def add_assistant_message(self, answer: str, sources: Optional[List[str]] = None):
        """Ajoute une réponse assistant."""
        self.add_message("assistant", answer, sources if sources else [])

    def get_history(self, last_n: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Récupère l'historique des messages.

        Args:
            last_n: Nombre de messages à retourner (None = tous)

        Returns:
            Liste des messages
        """
        if last_n:
            return self.messages[-last_n:]
        return self.messages

    def get_formatted_history(self, last_n: Optional[int] = None) -> str:
        """
        Formate l'historique pour inclusion dans un prompt.

        Args:
            last_n: Nombre de messages à formater

        Returns:
            Chaîne formatée de l'historique
        """
        messages = self.get_history(last_n)
        formatted = []

        for msg in messages:
            if msg["role"] == "user":
                formatted.append(f"Utilisateur: {msg['content']}")
            else:
                formatted.append(f"Assistant: {msg['content']}")

        return "\n".join(formatted)

    def format_history(self) -> str:
        """
        Formate l'historique complet pour affichage.

        Returns:
            Chaîne formatée de l'historique
        """
        return self.get_formatted_history()

    def get_recent_questions(self, n: int = 2) -> List[str]:
        """
        Récupère les n dernières questions posées.

        Args:
            n: Nombre de questions à récupérer

        Returns:
            Liste des questions récentes
        """
        questions = []
        for msg in reversed(self.messages):
            if msg["role"] == "user":
                questions.append(msg["content"])
                if len(questions) >= n:
                    break
        return list(reversed(questions))

    def get_summary(self) -> str:
        """
        Génère un résumé de la conversation.

        Returns:
            Résumé textuel de la conversation
        """
        return self.get_conversation_summary()

    def get_context_for_prompt(self, max_chars: int = 2000) -> str:
        """
        Récupère l'historique formaté en limitant la taille pour le prompt.

        Args:
            max_chars: Nombre maximum de caractères (approximation)

        Returns:
            Historique formaté limité
        """
        formatted = self.get_formatted_history()
        
        if len(formatted) <= max_chars:
            return formatted
        
        # Tronquer si trop long (garder les messages les plus récents)
        return formatted[-max_chars:]

    def get_context_summary(self, max_pairs: int = 3) -> str:
        """
        Génère un résumé de l'historique pour le contexte.

        Args:
            max_pairs: Nombre maximum de paires question-réponse à inclure

        Returns:
            Résumé formaté de l'historique
        """
        if not self.messages:
            return ""
        
        # Récupérer les dernières paires
        recent_messages = self.messages[-(max_pairs * 2):]
        
        summary_parts = []
        for msg in recent_messages:
            if msg["role"] == "user":
                summary_parts.append(f"Question: {msg['content']}")
            else:
                summary_parts.append(f"Réponse: {msg['content'][:200]}...")  # Limiter la longueur des réponses
        
        return "\n".join(summary_parts)

    def clear(self):
        """Efface tout l'historique."""
        self.messages = []

    def get_last_question(self) -> Optional[str]:
        """Récupère la dernière question posée."""
        for msg in reversed(self.messages):
            if msg["role"] == "user":
                return msg["content"]
        return None

    def get_last_answer(self) -> Optional[str]:
        """Récupère la dernière réponse donnée."""
        for msg in reversed(self.messages):
            if msg["role"] == "assistant":
                return msg["content"]
        return None

    def get_conversation_summary(self) -> str:
        """
        Génère un résumé de la conversation.

        Returns:
            Résumé textuel de la conversation
        """
        if not self.messages:
            return "Aucune conversation en cours."

        user_questions = [msg["content"] for msg in self.messages if msg["role"] == "user"]
        assistant_answers = [msg["content"] for msg in self.messages if msg["role"] == "assistant"]

        return f"""
Session: {self.session_id}
Questions posées: {len(user_questions)}
Réponses données: {len(assistant_answers)}
Dernière question: {user_questions[-1] if user_questions else 'N/A'}
        """.strip()

    def save_to_file(self, filepath: str):
        """
        Sauvegarde l'historique dans un fichier JSON.

        Args:
            filepath: Chemin du fichier de sauvegarde
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                "session_id": self.session_id,
                "messages": self.messages
            }, f, ensure_ascii=False, indent=2)

    def load_from_file(self, filepath: str):
        """
        Charge l'historique depuis un fichier JSON.

        Args:
            filepath: Chemin du fichier à charger
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.session_id = data.get("session_id", self.session_id)
            self.messages = data.get("messages", [])

    def __len__(self) -> int:
        """Nombre de messages dans l'historique."""
        return len(self.messages)

    def __repr__(self) -> str:
        return f"ConversationHistory(session_id={self.session_id}, messages={len(self.messages)})"


class ConversationMemory:
    """
    Gestionnaire de mémoire pour plusieurs sessions de conversation.
    """

    def __init__(self, max_sessions: int = 5):
        """
        Initialise la mémoire de conversation.

        Args:
            max_sessions: Nombre maximum de sessions à conserver
        """
        self.max_sessions = max_sessions
        self.sessions: Dict[str, ConversationHistory] = {}
        self.current_session: Optional[str] = None

    def create_session(self, session_id: Optional[str] = None) -> str:
        """
        Crée une nouvelle session de conversation.

        Args:
            session_id: ID de session personnalisé (auto-généré si None)

        Returns:
            ID de la session créée
        """
        if not session_id:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.sessions[session_id] = ConversationHistory()
        self.current_session = session_id
        return session_id

    def get_current_session(self) -> Optional[ConversationHistory]:
        """Récupère la session actuelle."""
        if self.current_session and self.current_session in self.sessions:
            return self.sessions[self.current_session]
        return None

    def switch_session(self, session_id: str) -> bool:
        """
        Change de session active.

        Args:
            session_id: ID de la session à activer

        Returns:
            True si succès, False si session inexistante
        """
        if session_id in self.sessions:
            self.current_session = session_id
            return True
        return False

    def delete_session(self, session_id: str):
        """Supprime une session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            if self.current_session == session_id:
                self.current_session = None

    def list_sessions(self) -> List[str]:
        """Liste les IDs des sessions disponibles."""
        return list(self.sessions.keys())

    def __repr__(self) -> str:
        return f"ConversationMemory(sessions={len(self.sessions)}, current={self.current_session})"
