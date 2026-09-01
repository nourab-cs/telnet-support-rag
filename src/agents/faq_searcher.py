"""
Agent 3 - Recherche FAQ

Rôle:
- Recherche uniquement dans la FAQ
- Fournit des réponses directes aux questions courantes
"""

from typing import List, Dict, Any
from langchain_core.documents import Document


class FAQSearcher:
    """
    Agent IA spécialisé dans la recherche FAQ.
    """

    def __init__(self, retriever, k: int = 3):
        """
        Initialise le chercheur FAQ.

        Args:
            retriever: Retriever LangChain pour la recherche sémantique
            k: Nombre de réponses FAQ à récupérer (default: 3)
        """
        self.retriever = retriever
        self.k = k

    def search(self, query: str) -> List[Document]:
        """
        Recherche dans la FAQ.

        Args:
            query: Question de l'utilisateur

        Returns:
            Liste de réponses FAQ pertinentes
        """
        # Construire une requête optimisée pour FAQ
        faq_query = f"FAQ: {query}"

        # Recherche sémantique
        documents = self.retriever.invoke(faq_query)

        # Filtrer uniquement les documents FAQ
        faq_documents = self._filter_faq(documents)

        return faq_documents[:self.k]

    def _filter_faq(self, documents: List[Document]) -> List[Document]:
        """
        Filtre pour ne garder que les documents FAQ.

        Args:
            documents: Liste de documents

        Returns:
            Documents FAQ uniquement
        """
        faq_docs = []
        for doc in documents:
            source = doc.metadata.get("source", "").lower()
            # Filtrer par nom de fichier contenant "faq"
            if "faq" in source or "problèmes_courants" in source:
                faq_docs.append(doc)

        return faq_docs

    def get_direct_answer(self, query: str) -> str:
        """
        Tente de trouver une réponse directe dans la FAQ.

        Args:
            query: Question de l'utilisateur

        Returns:
            Réponse directe ou message si non trouvée
        """
        faq_docs = self.search(query)

        if faq_docs:
            # Retourner la réponse la plus pertinente
            return faq_docs[0].page_content
        else:
            return "Aucune réponse directe trouvée dans la FAQ."
