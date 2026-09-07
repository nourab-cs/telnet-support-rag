"""TELNET Support Bot - point d'entrée principal."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Permet d'exécuter main.py directement depuis la racine du projet.
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag.pipeline import RAGPipeline  # noqa: E402


DATA_DIR = PROJECT_ROOT / "data"
DB_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "telnet_support"


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def print_result(result: dict) -> None:
    print("\nRéponse :")
    print(result.get("answer", "Pas de réponse générée"))

    documents = result.get("source_documents", [])
    if documents:
        print(f"\nSources utilisées : {len(documents)}")
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "inconnu")
            print(f"   {i}. {source}")

    scores = result.get("retrieval_scores", [])
    if scores:
        print(f"\nScores de pertinence : {[f'{score:.2f}' for score in scores]}")

    print("-" * 70)


def main() -> None:
    configure_logging()

    print("Initialisation du système RAG TELNET Support Bot")
    print("=" * 70)

    pipeline = RAGPipeline(
        data_dir=str(DATA_DIR),
        db_dir=str(DB_DIR),
        collection_name=COLLECTION_NAME,
        use_history=True,
        max_history=10,
    )

    try:
        pipeline.load_existing()
    except (FileNotFoundError, ValueError) as exc:
        print(f"Index non disponible ou incompatible : {exc}")
        print("Création d'un nouvel index...")
        pipeline.build_index()

    print("Mode conversationnel actif avec historique")
    print("Commandes disponibles :")
    print("  - 'quit' : quitter")
    print("=" * 70)

    while True:
        try:
            question = input("\nVotre question : ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAu revoir !")
            break

        command = question.lower()

        if command in {"quit", "exit", "q"}:
            print("Au revoir !")
            break

        if not question:
            continue

        try:
            print_result(pipeline.ask(question))
        except Exception as exc:
            logging.getLogger(__name__).exception("Erreur pendant le traitement de la question")
            print(f"Erreur : {exc}")


if __name__ == "__main__":
    main()
