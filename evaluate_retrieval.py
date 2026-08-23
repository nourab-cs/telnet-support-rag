"""
Script d'evaluation de la partie retrieval du systeme RAG
Teste uniquement la qualite de la recherche documentaire (sans LLM)
"""
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Ajouter le dossier src au path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pipeline import RAGPipeline


class RetrievalEvaluator:
    """Evaluateur de la retrieval RAG (sans generation)"""
    
    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline
        
        # Questions de test avec sources attendues
        self.test_questions = [
            {
                "question": "Comment installer SmartConnect ?",
                "expected_source": "01_guide_installation_smartconnect.md",
                "expected_keywords": ["installer", "dependances", "python", "pip", "git"]
            },
            {
                "question": "Quels sont les rôles utilisateurs disponibles ?",
                "expected_source": "04_guide_utilisateur_dashboard.md",
                "expected_keywords": ["admin", "utilisateur", "role", "permission"]
            },
            {
                "question": "Comment configurer MQTT ?",
                "expected_source": "07_configuration_reseau.md",
                "expected_keywords": ["mqtt", "broker", "port", "topic", "configuration"]
            },
            {
                "question": "Le service ne démarre pas, que faire ?",
                "expected_source": "08_troubleshooting_guides.md",
                "expected_keywords": ["service", "demarrer", "erreur", "log", "diagnostic"]
            },
            {
                "question": "Quelles sont les nouveautés de la version 3.3.0 ?",
                "expected_source": "06_notes_version.md",
                "expected_keywords": ["version", "3.3.0", "nouveaute", "mise", "jour"]
            }
        ]
    
    def evaluate_retrieval(self, question: str, expected_source: str, expected_keywords: List[str]) -> Dict[str, Any]:
        """Evalue la qualite de la retrieval"""
        start_time = time.time()
        
        # Utiliser directement le retriever
        sources = self.pipeline.generator.retriever.invoke(question)
        
        retrieval_time = time.time() - start_time
        
        # Extraire les informations des sources
        source_files = [doc.metadata.get("source", "") for doc in sources]
        source_contents = [doc.page_content for doc in sources]
        
        # Verifier si la source attendue est dans les resultats
        expected_found = any(expected_source in source for source in source_files)
        
        # Verifier les mots-cles dans le contenu
        all_content = " ".join(source_contents).lower()
        keywords_found = [kw for kw in expected_keywords if kw.lower() in all_content]
        
        return {
            "retrieval_time": retrieval_time,
            "expected_found": expected_found,
            "num_sources": len(sources),
            "source_files": source_files,
            "keywords_found": keywords_found,
            "keywords_found_count": len(keywords_found),
            "keywords_total": len(expected_keywords),
            "keyword_coverage": len(keywords_found) / len(expected_keywords) if expected_keywords else 0,
            "avg_chunk_length": sum(len(content) for content in source_contents) / len(source_contents) if source_contents else 0
        }
    
    def run_evaluation(self) -> Dict[str, Any]:
        """Execute l'evaluation de la retrieval"""
        print("=" * 70)
        print("EVALUATION DE LA RETRIEVAL RAG (sans LLM)")
        print("=" * 70)
        
        results = {
            "total_questions": len(self.test_questions),
            "evaluations": [],
            "timing": [],
            "metrics": {
                "expected_source_found": 0,
                "total_keyword_coverage": 0,
                "total_questions": 0
            }
        }
        
        for i, test_case in enumerate(self.test_questions, 1):
            question = test_case["question"]
            expected_source = test_case["expected_source"]
            expected_keywords = test_case["expected_keywords"]
            
            print(f"\n--- Question {i}/{len(self.test_questions)} ---")
            print(f"Question: {question}")
            print(f"Source attendue: {expected_source}")
            
            # Evaluer la retrieval
            eval_result = self.evaluate_retrieval(question, expected_source, expected_keywords)
            
            print(f"Temps de retrieval: {eval_result['retrieval_time']:.3f}s")
            print(f"Source attendue trouvee: {eval_result['expected_found']}")
            print(f"Nombre de documents retournes: {eval_result['num_sources']}")
            print(f"Mots-cles trouves: {eval_result['keywords_found_count']}/{eval_result['keywords_total']}")
            print(f"Couverture des mots-cles: {eval_result['keyword_coverage']:.1%}")
            print(f"Taille moyenne des chunks: {eval_result['avg_chunk_length']:.0f} caracteres")
            
            # Afficher les sources trouvees
            print(f"Sources retournees:")
            for j, source in enumerate(eval_result['source_files'], 1):
                match = "X" if expected_source in source else " "
                print(f"   {match} {j}. {source}")
            
            # Enregistrer les resultats
            results["evaluations"].append({
                "question": question,
                "expected_source": expected_source,
                "retrieval_time": eval_result['retrieval_time'],
                "evaluation": eval_result
            })
            
            results["timing"].append(eval_result['retrieval_time'])
            results["metrics"]["expected_source_found"] += eval_result['expected_found']
            results["metrics"]["total_keyword_coverage"] += eval_result['keyword_coverage']
            results["metrics"]["total_questions"] += 1
            
            print("-" * 70)
        
        # Calculer les statistiques globales
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: Dict[str, Any]):
        """Affiche le resume de l'evaluation"""
        print("\n" + "=" * 70)
        print("RESUME DE L'EVALUATION RETRIEVAL")
        print("=" * 70)
        
        # Statistiques de temps
        timing = results["timing"]
        avg_time = sum(timing) / len(timing) if timing else 0
        min_time = min(timing) if timing else 0
        max_time = max(timing) if timing else 0
        
        print(f"\n--- Performance Temporelle ---")
        print(f"Temps moyen de retrieval: {avg_time:.3f}s")
        print(f"Temps minimum: {min_time:.3f}s")
        print(f"Temps maximum: {max_time:.3f}s")
        
        # Statistiques de precision
        metrics = results["metrics"]
        source_accuracy = metrics["expected_source_found"] / metrics["total_questions"] if metrics["total_questions"] > 0 else 0
        avg_keyword_coverage = metrics["total_keyword_coverage"] / metrics["total_questions"] if metrics["total_questions"] > 0 else 0
        
        print(f"\n--- Qualite de la Retrieval ---")
        print(f"Precision de la source: {source_accuracy:.1%} ({metrics['expected_source_found']}/{metrics['total_questions']})")
        print(f"Couverture moyenne des mots-cles: {avg_keyword_coverage:.1%}")
        
        # Statistiques sur les resultats
        num_sources = [e["evaluation"]["num_sources"] for e in results["evaluations"]]
        avg_sources = sum(num_sources) / len(num_sources) if num_sources else 0
        
        print(f"\n--- Statistiques des Resultats ---")
        print(f"Nombre moyen de documents retournes: {avg_sources:.1f}")
        
        # Score global
        overall_score = (source_accuracy + avg_keyword_coverage) / 2
        
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
    
    print("Initialisation du pipeline RAG pour evaluation retrieval...")
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
    evaluator = RetrievalEvaluator(pipeline)
    
    # Execution de l'evaluation
    results = evaluator.run_evaluation()
    
    print("\nEvaluation terminee avec succès!")


if __name__ == "__main__":
    main()