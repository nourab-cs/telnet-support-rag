"""
Script d'evaluation du systeme RAG TELNET Support Bot
Mesure les performances : temps de reponse, qualite des reponses, relevance
"""
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Ajouter le dossier src au path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pipeline import RAGPipeline


class RAGEvaluator:
    """Evaluateur du systeme RAG"""
    
    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline
        
        # Questions de test avec reponses attendues
        self.test_questions = [
            {
                "question": "Comment installer SmartConnect ?",
                "expected_keywords": ["installer", "dependances", "python", "pip", "git"],
                "expected_source": "01_guide_installation_smartconnect.md"
            },
            {
                "question": "Quels sont les rôles utilisateurs disponibles ?",
                "expected_keywords": ["admin", "utilisateur", "role", "permission"],
                "expected_source": "04_guide_utilisateur_dashboard.md"
            },
            {
                "question": "Comment configurer MQTT ?",
                "expected_keywords": ["mqtt", "broker", "port", "topic", "configuration"],
                "expected_source": "07_configuration_reseau.md"
            },
            {
                "question": "Le service ne démarre pas, que faire ?",
                "expected_keywords": ["service", "demarrer", "erreur", "log", "diagnostic"],
                "expected_source": "08_troubleshooting_guides.md"
            },
            {
                "question": "Quelles sont les nouveautés de la version 3.3.0 ?",
                "expected_keywords": ["version", "3.3.0", "nouveaute", "mise", "jour"],
                "expected_source": "06_notes_version.md"
            }
        ]
    
    def evaluate_retrieval(self, question: str, expected_source: str) -> Dict[str, Any]:
        """Evalue la qualite de la retrieval"""
        result = self.pipeline.ask(question)
        
        sources = result.get("source_documents", [])
        source_files = [doc.metadata.get("source", "") for doc in sources]
        
        # Verifier si la source attendue est dans les resultats
        expected_found = any(expected_source in source for source in source_files)
        
        return {
            "expected_found": expected_found,
            "num_sources": len(sources),
            "source_files": source_files
        }
    
    def evaluate_generation(self, answer: str, expected_keywords: List[str]) -> Dict[str, Any]:
        """Evalue la qualite de la generation"""
        answer_lower = answer.lower()
        
        # Compter les mots-cles trouves
        keywords_found = [kw for kw in expected_keywords if kw.lower() in answer_lower]
        
        return {
            "keywords_found": keywords_found,
            "keywords_found_count": len(keywords_found),
            "keywords_total": len(expected_keywords),
            "keyword_coverage": len(keywords_found) / len(expected_keywords) if expected_keywords else 0
        }
    
    def run_full_evaluation(self) -> Dict[str, Any]:
        """Execute l'evaluation complete"""
        print("=" * 70)
        print("EVALUATION DU SYSTEME RAG")
        print("=" * 70)
        
        results = {
            "total_questions": len(self.test_questions),
            "evaluations": [],
            "timing": [],
            "retrieval_metrics": {
                "expected_source_found": 0,
                "total": 0
            },
            "generation_metrics": {
                "total_keyword_coverage": 0,
                "total_questions": 0
            }
        }
        
        for i, test_case in enumerate(self.test_questions, 1):
            question = test_case["question"]
            expected_keywords = test_case["expected_keywords"]
            expected_source = test_case["expected_source"]
            
            print(f"\n--- Question {i}/{len(self.test_questions)} ---")
            print(f"Question: {question}")
            
            # Mesurer le temps de reponse
            start_time = time.time()
            result = self.pipeline.ask(question)
            response_time = time.time() - start_time
            
            answer = result.get("result", "")
            
            print(f"Temps de reponse: {response_time:.2f}s")
            print(f"Longueur de la reponse: {len(answer)} caracteres")
            
            # Evaluer la retrieval
            retrieval_eval = self.evaluate_retrieval(question, expected_source)
            print(f"Source attendue trouvee: {retrieval_eval['expected_found']}")
            print(f"Nombre de sources: {retrieval_eval['num_sources']}")
            
            # Evaluer la generation
            generation_eval = self.evaluate_generation(answer, expected_keywords)
            print(f"Mots-cles trouves: {generation_eval['keywords_found_count']}/{generation_eval['keywords_total']}")
            print(f"Couverture des mots-cles: {generation_eval['keyword_coverage']:.1%}")
            
            # Enregistrer les resultats
            results["evaluations"].append({
                "question": question,
                "response_time": response_time,
                "answer_length": len(answer),
                "retrieval": retrieval_eval,
                "generation": generation_eval
            })
            
            results["timing"].append(response_time)
            results["retrieval_metrics"]["expected_source_found"] += retrieval_eval["expected_found"]
            results["retrieval_metrics"]["total"] += 1
            results["generation_metrics"]["total_keyword_coverage"] += generation_eval["keyword_coverage"]
            results["generation_metrics"]["total_questions"] += 1
            
            print("-" * 70)
        
        # Calculer les statistiques globales
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: Dict[str, Any]):
        """Affiche le resume de l'evaluation"""
        print("\n" + "=" * 70)
        print("RESUME DE L'EVALUATION")
        print("=" * 70)
        
        # Statistiques de temps
        timing = results["timing"]
        avg_time = sum(timing) / len(timing) if timing else 0
        min_time = min(timing) if timing else 0
        max_time = max(timing) if timing else 0
        
        print(f"\n--- Performance Temporelle ---")
        print(f"Temps moyen de reponse: {avg_time:.2f}s")
        print(f"Temps minimum: {min_time:.2f}s")
        print(f"Temps maximum: {max_time:.2f}s")
        
        # Statistiques de retrieval
        retrieval = results["retrieval_metrics"]
        retrieval_accuracy = retrieval["expected_source_found"] / retrieval["total"] if retrieval["total"] > 0 else 0
        
        print(f"\n--- Performance Retrieval ---")
        print(f"Precision de la source: {retrieval_accuracy:.1%} ({retrieval['expected_source_found']}/{retrieval['total']})")
        
        # Statistiques de generation
        generation = results["generation_metrics"]
        avg_keyword_coverage = generation["total_keyword_coverage"] / generation["total_questions"] if generation["total_questions"] > 0 else 0
        
        print(f"\n--- Performance Generation ---")
        print(f"Couverture moyenne des mots-cles: {avg_keyword_coverage:.1%}")
        
        # Evaluation globale
        overall_score = (retrieval_accuracy + avg_keyword_coverage) / 2
        
        print(f"\n--- Score Global ---")
        print(f"Score global: {overall_score:.1%}")
        
        if overall_score >= 0.8:
            print("Performance: EXCELLENTE")
        elif overall_score >= 0.6:
            print("Performance: BONNE")
        elif overall_score >= 0.4:
            print("Performance: MOYENNE")
        else:
            print("Performance: A AMELIORER")
        
        print("=" * 70)


def main():
    """Fonction principale"""
    # Configuration
    DATA_DIR = "./data"
    DB_DIR = "./chroma_db"
    COLLECTION_NAME = "telnet_support"
    
    print("Initialisation du pipeline RAG pour evaluation...")
    print("=" * 70)
    
    # Creation du pipeline
    pipeline = RAGPipeline(
        data_dir=DATA_DIR,
        db_dir=DB_DIR,
        collection_name=COLLECTION_NAME
    )
    
    # Construction du pipeline
    pipeline.build()
    
    # Creation de l'evaluateur
    evaluator = RAGEvaluator(pipeline)
    
    # Execution de l'evaluation
    results = evaluator.run_full_evaluation()
    
    print("\nEvaluation terminee avec succes!")


if __name__ == "__main__":
    main()