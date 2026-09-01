"""
Agent 5 - Génération de Réponse

Rôle:
- Fusionne toutes les informations collectées
- Génère une proposition de réponse contextualisée
- Inclut les sources utilisées
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict, Any
from langchain_core.documents import Document


class ResponseGenerator:
    """
    Agent IA pour la génération de réponses.
    """

    def __init__(self, model_name: str = "mistral", temperature: float = 0.1):
        """
        Initialise le générateur de réponses.

        Args:
            model_name: Nom du modèle LLM (default: mistral)
            temperature: Température pour la génération (default: 0.1)
        """
        self.llm = ChatOllama(model=model_name, temperature=temperature)
        self.parser = StrOutputParser()

    def generate(
        self,
        query: str,
        ticket_analysis: Dict[str, Any],
        document_results: List[Document],
        faq_results: List[Document],
        similar_tickets: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Génère une réponse en fusionnant toutes les informations.

        Args:
            query: Question originale
            ticket_analysis: Analyse du ticket (Agent 1)
            document_results: Résultats de recherche documentaire (Agent 2)
            faq_results: Résultats de recherche FAQ (Agent 3)
            similar_tickets: Tickets similaires (Agent 4)

        Returns:
            Dictionnaire avec la réponse et les métadonnées
        """
        # Construire le contexte
        context = self._build_context(
            document_results,
            faq_results,
            similar_tickets
        )

        # Construire le prompt
        prompt = self._build_prompt(
            query,
            ticket_analysis,
            context
        )

        # Générer la réponse
        response = self.llm.invoke(prompt)
        generated_text = response.content

        # Collecter les sources
        sources = self._collect_sources(document_results, faq_results)

        return {
            "response": generated_text,
            "sources": sources,
            "ticket_analysis": ticket_analysis,
            "context_used": {
                "documents": len(document_results),
                "faq": len(faq_results),
                "similar_tickets": len(similar_tickets) if similar_tickets else 0
            }
        }

    def _build_context(
        self,
        document_results: List[Document],
        faq_results: List[Document],
        similar_tickets: List[Dict[str, Any]] = None
    ) -> str:
        """
        Construit le contexte à partir des résultats des agents.

        Args:
            document_results: Résultats documentaires
            faq_results: Résultats FAQ
            similar_tickets: Tickets similaires

        Returns:
            Contexte formaté
        """
        context_parts = []

        # Documentation
        if document_results:
            context_parts.append("=== DOCUMENTATION ===")
            for i, doc in enumerate(document_results[:3], 1):
                context_parts.append(f"{i}. {doc.page_content[:300]}...")

        # FAQ
        if faq_results:
            context_parts.append("\n=== FAQ ===")
            for i, doc in enumerate(faq_results[:2], 1):
                context_parts.append(f"{i}. {doc.page_content[:200]}...")

        # Tickets similaires
        if similar_tickets:
            context_parts.append("\n=== TICKETS SIMILAIRES ===")
            for i, ticket_info in enumerate(similar_tickets[:2], 1):
                ticket = ticket_info["ticket"]
                resolution = ticket.get("resolution", "Pas de résolution")
                context_parts.append(f"{i}. Résolution similaire: {resolution[:200]}...")

        return "\n".join(context_parts)

    def _build_prompt(
        self,
        query: str,
        ticket_analysis: Dict[str, Any],
        context: str
    ) -> str:
        """
        Construit le prompt pour la génération.

        Args:
            query: Question originale
            ticket_analysis: Analyse du ticket
            context: Contexte construit

        Returns:
            Prompt formaté
        """
        prompt = f"""
Tu es un assistant technique pour TELNET SmartConnect.

INSTRUCTIONS:
- Génère une réponse CONCISE et DIRECTE (max 3-4 phrases)
- Utilise UNIQUEMENT les informations du contexte fourni
- Sois factuel et précis
- Si l'information n'est pas présente, dis-le clairement
- Ne jamais inventer d'informations

ANALYSE DU TICKET:
- Problème: {ticket_analysis.get('problem', 'N/A')}
- Domaine: {ticket_analysis.get('domain', 'N/A')}
- Priorité: {ticket_analysis.get('priority', 'N/A')}

CONTEXTE:
{context}

QUESTION:
{query}

RÉPONSE CONCISE EN FRANÇAIS:
"""
        return prompt

    def _collect_sources(
        self,
        document_results: List[Document],
        faq_results: List[Document]
    ) -> List[str]:
        """
        Collecte les sources utilisées.

        Args:
            document_results: Résultats documentaires
            faq_results: Résultats FAQ

        Returns:
            Liste des sources
        """
        sources = []

        for doc in document_results:
            source = doc.metadata.get("source", "Source inconnue")
            if source not in sources:
                sources.append(source)

        for doc in faq_results:
            source = doc.metadata.get("source", "Source inconnue")
            if source not in sources:
                sources.append(source)

        return sources
