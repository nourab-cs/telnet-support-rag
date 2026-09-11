"""TELNET Support Bot - point d'entrée principal."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# PATH PROJET
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parent

SRC_DIR = (
    PROJECT_ROOT / "src"
)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR),
    )


# ============================================================
# IMPORT
# ============================================================

from rag_projet_corrige.rag_fixed_final.pipeline import (  
    RAGPipeline,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = (
    PROJECT_ROOT / "data"
)

DB_DIR = (
    PROJECT_ROOT / "chroma_db"
)

COLLECTION_NAME = (
    "telnet_support"
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
)


# ============================================================
# AFFICHAGE RESULTAT
# ============================================================

def print_result(
    result: dict,
) -> None:

    print("\nRéponse :")
    print(
        result.get(
            "answer",
            "Pas de réponse générée",
        )
    )

    # --------------------------------------------------------
    # SOURCES
    # --------------------------------------------------------

    documents = result.get(
        "documents",
        [],
    )

    if documents:

        print(
            f"\nSources utilisées : {len(documents)}"
        )

        for i, document in enumerate(
            documents,
            1,
        ):

            metadata = (
                document.metadata or {}
            )

            source = (
                metadata.get("source")
                or metadata.get("filename")
                or "inconnu"
            )

            print(
                f"   {i}. {source}"
            )

    # --------------------------------------------------------
    # SCORES
    # --------------------------------------------------------

    scores = result.get(
        "scores",
        [],
    )

    score_type = result.get(
        "retrieval_score_type",
        "unknown",
    )

    if scores:

        print(
            f"\nScores ({score_type}) : "
            f"{[f'{score:.6f}' for score in scores]}"
        )

    # --------------------------------------------------------
    # SEARCH QUERY
    # --------------------------------------------------------

    search_query = result.get(
        "search_query"
    )

    if search_query:

        print(
            f"\nRequête de recherche : "
            f"{search_query}"
        )

    # --------------------------------------------------------
    # RETRIEVAL COUNT
    # --------------------------------------------------------

    retrieved_count = result.get(
        "retrieved_count",
        0,
    )

    relevant_count = result.get(
        "relevant_count",
        0,
    )

    print(
        "\nRetrieval : "
        f"{retrieved_count} résultat(s)"
    )

    print(
        "Pertinents : "
        f"{relevant_count} résultat(s)"
    )

    # --------------------------------------------------------
    # TRACE GATE
    # --------------------------------------------------------

    trace = result.get(
        "trace",
        {}
    )

    gate = trace.get(
        "relevance_gate",
        {}
    )

    if gate:

        threshold = gate.get(
            "threshold"
        )

        score_type = gate.get(
            "score_type",
            "unknown",
        )

        if threshold is None:

            print(
                f"Relevance Gate : "
                f"désactivé ({score_type})"
            )

        else:

            print(
                f"Relevance Gate : "
                f"threshold={threshold} "
                f"({score_type})"
            )

    print("-" * 70)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "Initialisation du système "
        "RAG TELNET Support Bot"
    )

    print("=" * 70)

    # ========================================================
    # PIPELINE
    # ========================================================

    pipeline = RAGPipeline(

        data_dir=str(
            DATA_DIR
        ),

        db_dir=str(
            DB_DIR
        ),

        collection_name=(
            COLLECTION_NAME
        ),

        embedding_model=(
            "BAAI/bge-m3"
        ),

        llm_model=(
            "mistral"
        ),

        # ----------------------------------------------------
        # RETRIEVAL
        # ----------------------------------------------------

        retrieval_type=(
            "hybrid"
        ),

        retrieval_k=8,

        retrieval_fetch_k=20,

        retrieval_lambda=0.6,

        hybrid_k=10,

        # ----------------------------------------------------
        # VECTOR RELEVANCE
        # ----------------------------------------------------

        # Utilisé pour :
        # similarity
        # mmr
        #
        # NON utilisé pour hybrid.
        # Désactivé pour hybrid car les scores RRF sont beaucoup plus bas (0.01-0.03)
        relevance_threshold=None,

        # ----------------------------------------------------
        # SIMILARITY SCORE THRESHOLD
        # ----------------------------------------------------

        similarity_score_threshold=0.40,

        # ----------------------------------------------------
        # CONTEXT
        # ----------------------------------------------------

        max_context_documents=6,

        # ----------------------------------------------------
        # HISTORY
        # ----------------------------------------------------

        use_history=True,

        max_history=10,

        max_history_chars=5000,

        # ----------------------------------------------------
        # QUERY REWRITING
        # ----------------------------------------------------

        enable_query_rewriting=True,
    )

    # ========================================================
    # LOAD INDEX
    # ========================================================

    try:

        pipeline.load_existing()

        print(
            "Index Chroma existant chargé."
        )

    except (
        FileNotFoundError,
        ValueError,
    ) as exc:

        print(
            f"Index non disponible ou incompatible : {exc}"
        )

        print(
            "Création d'un nouvel index..."
        )

        result = (
            pipeline.build_index()
        )

        print(
            "Index créé : "
            f"{result['documents']} document(s), "
            f"{result['chunks']} chunk(s), "
            f"{result['latency']:.2f}s"
        )

    # ========================================================
    # STATUS
    # ========================================================

    status = (
        pipeline.status()
    )

    print(
        f"Retrieval : "
        f"{status['retrieval_type']}"
    )

    print(
        f"K : "
        f"{status['retrieval_k']}"
    )

    print(
        f"Hybrid K : "
        f"{status['hybrid_k']}"
    )

    print(
        "Mode conversationnel actif "
        "avec historique"
    )

    print(
        "QueryRewriter Mistral/Ollama actif"
    )

    # ========================================================
    # COMMANDES
    # ========================================================

    print(
        "Commandes disponibles :"
    )

    print(
        "  - 'quit' : quitter"
    )

    print("=" * 70)

    # ========================================================
    # BOUCLE
    # ========================================================

    while True:

        try:

            question = input(
                "\nVotre question : "
            ).strip()

        except (
            EOFError,
            KeyboardInterrupt,
        ):

            print(
                "\nAu revoir !"
            )

            break

        command = (
            question.lower()
        )

        if command in {
            "quit",
            "exit",
            "q",
        }:

            print(
                "Au revoir !"
            )

            break

        if not question:
            continue

        try:

            result = (
                pipeline.ask(
                    question
                )
            )

            print_result(
                result
            )

        except Exception as exc:

            logger.exception(
                "Erreur pendant le traitement de la question."
            )

            print(
                f"\nErreur : {exc}"
            )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()

