"""
Agent 1 - Analyse du Ticket

Rôle:
- Identifier le problème
- Extraire les mots-clés
- Détecter le domaine concerné
- Déterminer la priorité
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from typing import Dict, Any


class TicketAnalyzer:
    """
    Agent IA pour l'analyse des tickets support.
    """

    def __init__(self, model_name: str = "mistral"):
        """
        Initialise l'analyseur de tickets.

        Args:
            model_name: Nom du modèle LLM (default: mistral)
        """
        self.llm = ChatOllama(model=model_name, temperature=0.1)
        self.parser = JsonOutputParser()

        # Prompt pour l'analyse de ticket
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """Tu es un expert en analyse de tickets support pour TELNET SmartConnect.

Analyse le ticket et retourne ta réponse au format JSON avec:
- problem: description courte du problème (max 20 mots)
- keywords: liste de 3-5 mots-clés pertinents
- domain: domaine technique concerné (réseau, installation, api, dashboard, maintenance, autre)
- priority: niveau de priorité (basse, moyenne, haute, critique)
- urgency: niveau d'urgence (1-5, où 5 est le plus urgent)
- reason: brève explication du choix de priorité"""),
            ("user", "Ticket:\n\n{ticket_content}")
        ])

        self.chain = self.analysis_prompt | self.llm | self.parser

    def analyze(self, ticket_content: str) -> Dict[str, Any]:
        """
        Analyse un ticket de support.

        Args:
            ticket_content: Contenu du ticket à analyser

        Returns:
            Dictionnaire avec l'analyse du ticket
        """
        try:
            result = self.chain.invoke({"ticket_content": ticket_content})
            return {
                "problem": result.get("problem", "Problème non identifié"),
                "keywords": result.get("keywords", []),
                "domain": result.get("domain", "autre"),
                "priority": result.get("priority", "moyenne"),
                "urgency": result.get("urgency", 3),
                "reason": result.get("reason", "Analyse standard")
            }
        except Exception as e:
            print(f"Erreur lors de l'analyse du ticket: {e}")
            # Retourner des valeurs par défaut
            return {
                "problem": "Problème non identifié",
                "keywords": [],
                "domain": "autre",
                "priority": "moyenne",
                "urgency": 3,
                "reason": "Erreur lors de l'analyse"
            }

    def extract_entities(self, ticket_content: str) -> Dict[str, list]:
        """
        Extrait les entités mentionnées dans le ticket.

        Args:
            ticket_content: Contenu du ticket

        Returns:
            Dictionnaire avec les entités trouvées
        """
        # Cette méthode peut être étendue avec NER
        entities = {
            "services": [],
            "components": [],
            "error_codes": [],
            "versions": []
        }

        # Extraction simple basée sur des patterns
        import re

        # Codes d'erreur typiques
        error_pattern = r'(?:error|erreur|code)\s*[:#]?\s*(\d{3,4}|[A-Z]{2,}\d+)'
        entities["error_codes"] = re.findall(error_pattern, ticket_content, re.IGNORECASE)

        # Versions
        version_pattern = r'(?:version|v)\s*[:#]?\s*(\d+\.\d+\.\d+)'
        entities["versions"] = re.findall(version_pattern, ticket_content, re.IGNORECASE)

        return entities
