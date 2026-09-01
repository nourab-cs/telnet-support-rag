"""
Agent 4 - Recherche Tickets Similaires

Rôle:
- Analyse les tickets déjà résolus
- Trouve des tickets similaires
- Suggère des solutions basées sur l'historique
"""

from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import numpy as np


class TicketSimilarity:
    """
    Agent IA pour la recherche de tickets similaires.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        """
        Initialise le chercheur de tickets similaires.

        Args:
            model_name: Modèle d'embedding pour la similarité
        """
        self.model = SentenceTransformer(model_name)
        self.ticket_history = []  # À remplacer par base de données PostgreSQL

    def add_ticket_to_history(self, ticket: Dict[str, Any]):
        """
        Ajoute un ticket à l'historique (pour simulation).

        Args:
            ticket: Dictionnaire avec les données du ticket
        """
        self.ticket_history.append(ticket)

    def find_similar_tickets(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Trouve des tickets similaires dans l'historique.

        Args:
            query: Description du problème actuel
            top_k: Nombre de tickets similaires à retourner

        Returns:
            Liste des tickets similaires avec scores de similarité
        """
        if not self.ticket_history:
            return []

        # Encoder la requête
        query_embedding = self.model.encode(query)

        # Encoder tous les tickets de l'historique
        ticket_texts = [ticket.get("description", "") for ticket in self.ticket_history]
        ticket_embeddings = self.model.encode(ticket_texts)

        # Calculer les similarités cosinus
        similarities = self._cosine_similarity(query_embedding, ticket_embeddings)

        # Trier par similarité
        sorted_indices = np.argsort(similarities)[::-1]

        # Retourner les top_k tickets les plus similaires
        similar_tickets = []
        for idx in sorted_indices[:top_k]:
            if similarities[idx] > 0.5:  # Seuil de similarité
                similar_tickets.append({
                    "ticket": self.ticket_history[idx],
                    "similarity_score": float(similarities[idx])
                })

        return similar_tickets

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
        """
        Calcule la similarité cosinus entre deux vecteurs.

        Args:
            vec1: Premier vecteur
            vec2: Deuxième vecteur (ou matrice)

        Returns:
            Scores de similarité
        """
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2, axis=1)
        return np.dot(vec1, vec2.T) / (norm1 * norm2)

    def get_suggested_solution(self, query: str) -> Optional[str]:
        """
        Suggère une solution basée sur les tickets similaires.

        Args:
            query: Description du problème actuel

        Returns:
            Solution suggérée ou None
        """
        similar_tickets = self.find_similar_tickets(query, top_k=1)

        if similar_tickets and similar_tickets[0]["similarity_score"] > 0.7:
            ticket = similar_tickets[0]["ticket"]
            return ticket.get("resolution", "Pas de solution disponible")
        else:
            return None
