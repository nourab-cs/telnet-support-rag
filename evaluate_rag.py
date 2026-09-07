"""
Script d'évaluation du système RAG TELNET Support Bot

Tests effectués :
1. Questions techniques simples
2. Questions complexes avec historique
3. Questions conversationnelles
4. Évaluation des scores de pertinence
5. Test des sources utilisées
"""
import sys
from pathlib import Path
import json
from datetime import datetime

# Ajouter le dossier src au path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag.pipeline import RAGPipeline


class RAGEvaluator:
    """Évaluateur du système RAG"""
    
    def __init__(self, data_dir="./data", db_dir="./chroma_db"):
        self.data_dir = data_dir
        self.db_dir = db_dir
        self.pipeline = None
        self.results = []
        
    def initialize(self):
        """Initialise le pipeline RAG"""
        print("Initialisation du pipeline RAG...")
        self.pipeline = RAGPipeline(
            data_dir=self.data_dir,
            db_dir=self.db_dir,
            collection_name="telnet_support",
            use_history=True,
            max_history=10
        )
        
        try:
            self.pipeline.load_existing()
            print("✓ Index existant chargé")
        except (FileNotFoundError, ValueError) as e:
            print(f"Index non disponible: {e}")
            print("Création d'un nouvel index...")
            self.pipeline.build_index()
            print("✓ Index créé")
    
    def test_question(self, question, expected_type="technical", expected_keywords=None):
        """Teste une question individuelle"""
        print(f"\n{'='*70}")
        print(f"Question: {question}")
        print(f"Type attendu: {expected_type}")
        print(f"{'='*70}")
        
        result = self.pipeline.ask(question)
        
        # Analyse du résultat
        analysis = {
            "question": question,
            "expected_type": expected_type,
            "actual_type": result.get("query_type", "unknown"),
            "answer": result.get("answer", ""),
            "sources_used": result.get("sources_used", 0),
            "retrieval_scores": result.get("retrieval_scores", []),
            "documents_retrieved": result.get("documents_retrieved", 0),
            "documents_relevant": result.get("documents_relevant", 0),
            "expected_keywords": expected_keywords,
            "timestamp": datetime.now().isoformat()
        }
        
        # Vérification des mots-clés attendus
        if expected_keywords:
            answer_lower = result.get("answer", "").lower()
            found_keywords = [kw for kw in expected_keywords if kw.lower() in answer_lower]
            analysis["keywords_found"] = found_keywords
            analysis["keywords_match"] = len(found_keywords) == len(expected_keywords)
        
        # Affichage des résultats
        print(f"Type détecté: {analysis['actual_type']}")
        print(f"Sources utilisées: {analysis['sources_used']}")
        print(f"Documents récupérés: {analysis['documents_retrieved']}")
        print(f"Documents pertinents: {analysis['documents_relevant']}")
        
        if analysis['retrieval_scores']:
            print(f"Scores de pertinence: {[f'{score:.3f}' for score in analysis['retrieval_scores']]}")
            avg_score = sum(analysis['retrieval_scores']) / len(analysis['retrieval_scores'])
            print(f"Score moyen: {avg_score:.3f}")
            analysis["avg_score"] = avg_score
        
        print(f"\nRéponse: {analysis['answer'][:200]}...")
        
        # Vérification du type
        type_match = analysis['actual_type'] == expected_type
        analysis["type_match"] = type_match
        print(f"✓ Type correct" if type_match else "✗ Type incorrect")
        
        if expected_keywords:
            print(f"Mots-clés trouvés: {found_keywords}/{expected_keywords}")
            print(f"✓ Mots-clés complets" if analysis['keywords_match'] else "✗ Mots-clés incomplets")
        
        self.results.append(analysis)
        return analysis
    
    def run_technical_tests(self):
        """Teste des questions techniques"""
        print("\n" + "="*70)
        print("TESTS TECHNIQUES")
        print("="*70)
        
        technical_questions = [
            {
                "question": "Comment installer TELNET SmartConnect ?",
                "expected_type": "technical",
                "expected_keywords": ["install", "télécharger", "configuration"]
            },
            {
                "question": "Quels sont les problèmes courants avec SmartConnect ?",
                "expected_type": "technical",
                "expected_keywords": ["problème", "erreur", "solution"]
            },
            {
                "question": "Comment configurer le réseau pour TELNET SmartConnect ?",
                "expected_type": "technical",
                "expected_keywords": ["réseau", "configuration", "port"]
            },
            {
                "question": "Quelle est la procédure de maintenance ?",
                "expected_type": "technical",
                "expected_keywords": ["maintenance", "procédure", "mise à jour"]
            },
            {
                "question": "Comment utiliser l'API TELNET SmartConnect ?",
                "expected_type": "technical",
                "expected_keywords": ["api", "endpoint", "authentification"]
            }
        ]
        
        for test in technical_questions:
            self.test_question(
                test["question"],
                test["expected_type"],
                test["expected_keywords"]
            )
    
    def run_conversational_tests(self):
        """Teste des questions conversationnelles"""
        print("\n" + "="*70)
        print("TESTS CONVERSATIONNELS")
        print("="*70)
        
        conversational_questions = [
            ("Bonjour", "conversation"),
            ("Merci", "conversation"),
            ("Au revoir", "conversation"),
            ("Salut", "conversation")
        ]
        
        for question, expected_type in conversational_questions:
            self.test_question(question, expected_type)
    
    def run_context_tests(self):
        """Teste les questions avec contexte historique"""
        print("\n" + "="*70)
        print("TESTS AVEC CONTEXTE")
        print("="*70)
        
        # Premier contexte
        self.test_question("Comment installer TELNET SmartConnect ?", "technical")
        
        # Question de suivi
        self.test_question("Et après l'installation ?", "technical", ["configuration", "paramètres"])
        
        # Autre contexte
        self.test_question("Quels sont les ports réseau utilisés ?", "technical", ["port", "réseau"])
        
        # Question de suivi
        self.test_question("Comment les configurer ?", "technical", ["configuration", "port"])
    
    def run_edge_cases(self):
        """Teste des cas limites"""
        print("\n" + "="*70)
        print("CAS LIMITES")
        print("="*70)
        
        # Question vide
        self.test_question("", "conversation")
        
        # Question hors sujet
        self.test_question("Comment faire un gâteau au chocolat ?", "technical")
        
        # Question très courte
        self.test_question("Erreur", "technical")
        
        # Question très longue
        long_question = " ".join(["Quelle est la procédure complète"] * 10)
        self.test_question(long_question, "technical")
    
    def generate_report(self):
        """Génère un rapport d'évaluation"""
        print("\n" + "="*70)
        print("RAPPORT D'ÉVALUATION")
        print("="*70)
        
        total_tests = len(self.results)
        technical_tests = [r for r in self.results if r["expected_type"] == "technical"]
        conversational_tests = [r for r in self.results if r["expected_type"] == "conversation"]
        
        # Statistiques de type
        type_matches = sum(1 for r in self.results if r.get("type_match", False))
        type_accuracy = (type_matches / total_tests * 100) if total_tests > 0 else 0
        
        # Statistiques de mots-clés
        keyword_tests = [r for r in self.results if r.get("expected_keywords")]
        keyword_matches = sum(1 for r in keyword_tests if r.get("keywords_match", False))
        keyword_accuracy = (keyword_matches / len(keyword_tests) * 100) if keyword_tests else 0
        
        # Statistiques de pertinence
        technical_with_scores = [r for r in technical_tests if r.get("retrieval_scores")]
        avg_scores = [r.get("avg_score", 0) for r in technical_with_scores]
        overall_avg_score = sum(avg_scores) / len(avg_scores) if avg_scores else 0
        
        # Sources utilisées
        avg_sources = sum(r.get("sources_used", 0) for r in technical_tests) / len(technical_tests) if technical_tests else 0
        
        print(f"\nTests totaux: {total_tests}")
        print(f"Tests techniques: {len(technical_tests)}")
        print(f"Tests conversationnels: {len(conversational_tests)}")
        print(f"\nPrécision du type: {type_accuracy:.1f}%")
        print(f"Précision des mots-clés: {keyword_accuracy:.1f}%")
        print(f"Score moyen de pertinence: {overall_avg_score:.3f}")
        print(f"Sources moyennes utilisées: {avg_sources:.1f}")
        
        # Détails par catégorie
        print(f"\n--- Tests techniques ---")
        for r in technical_tests:
            status = "✓" if r.get("type_match", False) else "✗"
            print(f"{status} {r['question'][:50]}... (score: {r.get('avg_score', 0):.3f})")
        
        print(f"\n--- Tests conversationnels ---")
        for r in conversational_tests:
            status = "✓" if r.get("type_match", False) else "✗"
            print(f"{status} {r['question']}")
        
        # Sauvegarde du rapport
        report = {
            "summary": {
                "total_tests": total_tests,
                "technical_tests": len(technical_tests),
                "conversational_tests": len(conversational_tests),
                "type_accuracy": type_accuracy,
                "keyword_accuracy": keyword_accuracy,
                "avg_relevance_score": overall_avg_score,
                "avg_sources_used": avg_sources
            },
            "detailed_results": self.results,
            "generated_at": datetime.now().isoformat()
        }
        
        report_path = Path(__file__).parent / "evaluation_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Rapport sauvegardé: {report_path}")
        
        return report
    
    def run_full_evaluation(self):
        """Exécute l'évaluation complète"""
        print("DÉBUT DE L'ÉVALUATION RAG")
        print("="*70)
        
        try:
            self.initialize()
            
            # Tests de base
            self.run_conversational_tests()
            self.run_technical_tests()
            
            # Tests avancés
            self.run_context_tests()
            self.run_edge_cases()
            
            # Rapport
            report = self.generate_report()
            
            print("\n" + "="*70)
            print("ÉVALUATION TERMINÉE")
            print("="*70)
            
            return report
            
        except Exception as e:
            print(f"\n✗ Erreur lors de l'évaluation: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """Point d'entrée principal"""
    evaluator = RAGEvaluator()
    evaluator.run_full_evaluation()


if __name__ == "__main__":
    main()
