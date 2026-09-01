"""
Agent 6 - Vérification de la Réponse

Rôle:
- Vérifie la cohérence de la réponse
- Contrôle la présence de sources
- Estime le niveau de confiance
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from typing import Dict, Any, List


class ResponseVerifier:
    """
    Agent IA pour la vérification des réponses.
    """

    def __init__(self, model_name: str = "mistral"):
        """
        Initialise le vérificateur de réponses.

        Args:
            model_name: Nom du modèle LLM (default: mistral)
        """
        self.llm = ChatOllama(model=model_name, temperature=0.1)
        self.parser = JsonOutputParser()

        # Prompt pour la vérification
        self.verification_prompt = ChatPromptTemplate.from_messages([
            ("system", """Tu es un expert en vérification de réponses techniques.

Vérifie la réponse et retourne ton analyse au format JSON avec:
- coherence: score de cohérence (0-1, où 1 est parfaitement cohérent)
- has_sources: true si des sources sont citées, false sinon
- confidence: niveau de confiance global (0-1)
- issues: liste des problèmes détectés (vide si aucun)
- recommendation: "accepter", "réviser" ou "regénérer"
- reason: explication de la décision"""),
            ("user", """QUESTION: {question}
RÉPONSE: {response}
SOURCES: {sources}
""")
        ])

        self.chain = self.verification_prompt | self.llm | self.parser

    def verify(
        self,
        question: str,
        response: str,
        sources: List[str],
        context_used: Dict[str, int]
    ) -> Dict[str, Any]:
        """
        Vérifie la qualité de la réponse.

        Args:
            question: Question originale
            response: Réponse générée
            sources: Sources utilisées
            context_used: Statistiques du contexte utilisé

        Returns:
            Dictionnaire avec les résultats de vérification
        """
        # Vérification heuristique
        heuristic_score = self._heuristic_verification(
            response,
            sources,
            context_used
        )

        # Vérification LLM
        try:
            llm_result = self.chain.invoke({
                "question": question,
                "response": response,
                "sources": ", ".join(sources) if sources else "Aucune"
            })

            # Combiner les scores
            final_confidence = (heuristic_score["confidence"] * 0.4) + (llm_result.get("confidence", 0.5) * 0.6)

            return {
                "coherence": llm_result.get("coherence", heuristic_score["coherence"]),
                "has_sources": len(sources) > 0,
                "confidence": final_confidence,
                "issues": llm_result.get("issues", []),
                "recommendation": llm_result.get("recommendation", "accepter"),
                "reason": llm_result.get("reason", "Vérification standard"),
                "heuristic_score": heuristic_score
            }
        except Exception as e:
            print(f"Erreur lors de la vérification LLM: {e}")
            # Retourner uniquement le score heuristique
            return heuristic_score

    def _heuristic_verification(
        self,
        response: str,
        sources: List[str],
        context_used: Dict[str, int]
    ) -> Dict[str, Any]:
        """
        Vérification heuristique de la réponse.

        Args:
            response: Réponse générée
            sources: Sources utilisées
            context_used: Statistiques du contexte

        Returns:
            Score heuristique
        """
        score = 0.5  # Score de base

        # Vérifier la longueur de la réponse
        if 50 <= len(response) <= 500:
            score += 0.1
        elif len(response) > 500:
            score -= 0.1

        # Vérifier la présence de sources
        if sources:
            score += 0.2
        else:
            score -= 0.3

        # Vérifier l'utilisation du contexte
        total_context = sum(context_used.values())
        if total_context > 0:
            score += 0.1

        # Vérifier les indicateurs de mauvaise qualité
        bad_indicators = ["je ne sais pas", "information non disponible", "pas dans la documentation"]
        if any(indicator in response.lower() for indicator in bad_indicators):
            score -= 0.2

        # Limiter le score entre 0 et 1
        score = max(0, min(1, score))

        return {
            "coherence": score,
            "has_sources": len(sources) > 0,
            "confidence": score,
            "issues": [],
            "recommendation": "accepter" if score > 0.6 else "réviser",
            "reason": "Vérification heuristique"
        }
