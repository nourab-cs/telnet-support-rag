"""TELNET Support Bot - point d'entrée principal."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

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

    documents = result.get("documents", [])

    if documents:

        print(f"\nSources utilisées : {len(documents)}")

        for i, doc in enumerate(documents, 1):

            metadata = doc.metadata or {}

            source = (
                metadata.get("source")
                or metadata.get("filename")
                or "inconnu"
            )

            print(f"   {i}. {source}")

    scores = result.get("scores", [])

    if scores:

        print(
            "\nScores de pertinence : "
            f"{[f'{score:.2f}' for score in scores]}"
        )

    search_query = result.get("search_query")

    if search_query:

        print(
            f"\nRequête de recherche : {search_query}"
        )

    trace = result.get("trace", {})

    if trace:

        latency = trace.get("latency", {})

        print(
            "\nLatence : "
            f"{latency.get('total', 0):.3f}s"
        )

    print("-" * 70)


def main() -> None:

    configure_logging()

    print(
        "Initialisation du système RAG TELNET Support Bot"
    )
    print("=" * 70)

    pipeline = RAGPipeline(
        data_dir=str(DATA_DIR),
        db_dir=str(DB_DIR),
        collection_name=COLLECTION_NAME,
        embedding_model="BAAI/bge-m3",
        llm_model="mistral",

        retrieval_type="mmr",
        retrieval_k=8,
        retrieval_fetch_k=20,
        retrieval_lambda=0.6,

        relevance_threshold=0.40,

        max_context_documents=6,

        use_history=True,
        max_history=10,

        enable_query_rewriting=True,
    )

    try:

        pipeline.load_existing()

    except (FileNotFoundError, ValueError) as exc:

        print(
            f"Index non disponible ou incompatible : {exc}"
        )

        print(
            "Création d'un nouvel index..."
        )

        pipeline.build_index()

    print(
        "Mode conversationnel actif avec historique"
    )

    print(
        "QueryRewriter Mistral/Ollama actif"
    )

    print("Commandes disponibles :")
    print("  - 'quit' : quitter")

    print("=" * 70)

    while True:

        try:

            question = input(
                "\nVotre question : "
            ).strip()

        except (EOFError, KeyboardInterrupt):

            print("\nAu revoir !")
            break

        command = question.lower()

        if command in {
            "quit",
            "exit",
            "q",
        }:

            print("Au revoir !")
            break

        if not question:
            continue

        try:

            result = pipeline.ask(question)

            print_result(result)

        except Exception as exc:

            logging.getLogger(
                __name__
            ).exception(
                "Erreur pendant le traitement de la question"
            )

            print(
                f"Erreur : {exc}"
            )


if __name__ == "__main__":
    main()