"""
Agent 2 - Recherche Documentaire

Rôle:
- Recherche dans la documentation technique
- Recherche dans les procédures
- Recherche dans les guides techniques
"""

from typing import List, Dict, Any
from langchain_core.documents import Document


class DocumentSearcher:
    """
    Agent IA pour la recherche documentaire.
    """

    def __init__(self, retriever, k: int = 4):
        """
        Initialise le chercheur documentaire.

        Args:
            retriever: Retriever LangChain pour la recherche sémantique
            k: Nombre de documents à récupérer (default: 4)
        """
        self.retriever = retriever
        self.k = k

    def search(self, query: str, domain: str = None) -> List[Document]:
        """
        Recherche des documents pertinents.

        Args:
            query: Requête de recherche
            domain: Domaine technique pour filtrer (optionnel)

        Returns:
            Liste de documents pertinents
        """
        # Recherche sémantique
        documents = self.retriever.invoke(query)

        # Filtrer par domaine si spécifié
        if domain:
            documents = self._filter_by_domain(documents, domain)

        # Limiter au nombre k
        return documents[:self.k]

    def _filter_by_domain(self, documents: List[Document], domain: str) -> List[Document]:
        """
        Filtre les documents par domaine technique.

        Args:
            documents: Liste de documents
            domain: Domaine à filtrer

        Returns:
            Documents filtrés
        """
        domain_keywords = {
            "réseau": ["réseau", "mqtt", "tcp", "ip", "configuration"],
            "installation": ["installation", "setup", "déploiement", "python"],
            "api": ["api", "endpoint", "rest", "http"],
            "dashboard": ["dashboard", "interface", "ui", "utilisateur"],
            "maintenance": ["maintenance", "procédure", "mise à jour"]
        }

        keywords = domain_keywords.get(domain.lower(), [])

        filtered = []
        for doc in documents:
            content = doc.page_content.lower()
            metadata_source = doc.metadata.get("source", "").lower()

            if any(keyword in content or keyword in metadata_source for keyword in keywords):
                filtered.append(doc)

        return filtered if filtered else documents[:self.k]

    def search_by_keywords(self, keywords: List[str]) -> List[Document]:
        """
        Recherche par mots-clés.

        Args:
            keywords: Liste de mots-clés

        Returns:
            Documents correspondants aux mots-clés
        """
        query = " ".join(keywords)
        return self.retriever.invoke(query)[:self.k]
