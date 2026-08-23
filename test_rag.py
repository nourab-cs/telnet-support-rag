"""
Script de test pour le systeme RAG TELNET Support Bot
"""
import sys
from pathlib import Path

# Ajouter le dossier src au path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pipeline import RAGPipeline


def test_rag():
    """Test le systeme RAG avec une question predefinie"""
    
    # Configuration
    DATA_DIR = "./data"
    DB_DIR = "./chroma_db"
    COLLECTION_NAME = "telnet_support"
    
    print("Test du systeme RAG TELNET Support Bot")
    print("=" * 70)
    
    # Creation du pipeline
    pipeline = RAGPipeline(
        data_dir=DATA_DIR,
        db_dir=DB_DIR,
        collection_name=COLLECTION_NAME
    )
    
    # Construction du pipeline
    pipeline.build()
    
    # Test avec une question
    test_question = "Comment installer SmartConnect ?"
    
    print(f"\nTest avec la question : {test_question}")
    print("-" * 70)
    
    try:
        result = pipeline.ask(test_question)
        
        print("\nReponse :")
        print(result.get("result", "Pas de reponse generee"))
        
        if "source_documents" in result:
            print("\nSources utilisees :")
            for i, doc in enumerate(result["source_documents"], 1):
                source = doc.metadata.get("source", "inconnu")
                print(f"   {i}. {source}")
        
        print("\n" + "=" * 70)
        print("Test termine avec succes !")
        
    except Exception as e:
        print(f"\nErreur lors du test : {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_rag()