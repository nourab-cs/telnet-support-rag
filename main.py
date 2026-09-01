"""
TELNET Support Bot - RAG System
Point d'entrée principal pour le système RAG fonctionnel
"""
import sys
from pathlib import Path

# Ajouter le dossier src au path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag.pipeline import RAGPipeline


def main():
    # Configuration
    DATA_DIR = "./data"
    DB_DIR = "./chroma_db"
    COLLECTION_NAME = "telnet_support"
    
    print("Initialisation du systeme RAG TELNET Support Bot")
    print("=" * 70)
    
    # Création du pipeline avec historique activé
    pipeline = RAGPipeline(
        data_dir=DATA_DIR,
        db_dir=DB_DIR,
        collection_name=COLLECTION_NAME,
        use_history=True,
        max_history=10
    )
    
    # Construction du pipeline
    pipeline.build()
    
    # Mode interactif avec commandes spéciales
    print("Mode conversationnel actif avec historique")
    print("Commandes disponibles :")
    print("  - 'quit' : quitter")
    print("  - 'history' : afficher l'historique")
    print("  - 'clear' : effacer l'historique")
    print("  - 'summary' : résumé de la conversation")
    print("=" * 70)
    
    while True:
        question = input("\nVotre question : ").strip()
        
        if question.lower() in ("quit", "exit", "q"):
            print("Au revoir !")
            break
        
        # Commandes spéciales
        if question.lower() == "history":
            history = pipeline.get_history()
            if history:
                print("\n--- Historique de conversation ---")
                print(history)
                print("--------------------------------")
            else:
                print("Aucun historique disponible")
            continue
        
        if question.lower() == "clear":
            pipeline.clear_history()
            print("Historique efface")
            continue
        
        if question.lower() == "summary":
            summary = pipeline.get_history_summary()
            print(f"\n--- Resume ---")
            print(summary)
            print("---------------")
            continue
        
        if not question:
            continue
        
        try:
            result = pipeline.ask(question)
            
            print("\nReponse :")
            print(result.get("result", "Pas de reponse generee"))
            
            if "source_documents" in result:
                print("\nSources :")
                for i, doc in enumerate(result["source_documents"], 1):
                    source = doc.metadata.get("source", "inconnu")
                    print(f"   {i}. {source}")
            
            print("-" * 70)
            
        except Exception as e:
            print(f"Erreur : {str(e)}")


if __name__ == "__main__":
    main()