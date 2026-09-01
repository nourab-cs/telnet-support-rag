from src.rag.document_loader import DocumentLoader
from src.rag.semantic_chunker import SemanticChunking


def main():

    # --------------------------------------------------
    # 1. Charger les documents
    # --------------------------------------------------

    loader = DocumentLoader("data")

    documents = loader.load()

    print("\n" + "=" * 60)
    print("DOCUMENTS CHARGÉS")
    print("=" * 60)

    print(f"Nombre de documents : {len(documents)}")





    # --------------------------------------------------
    # 3. Chunking sémantique
    # --------------------------------------------------

    chunker = SemanticChunking()

    chunks = chunker.chunk_documents(documents)


    # --------------------------------------------------
    # 4. Afficher les résultats
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("RÉSULTATS DU CHUNKING")
    print("=" * 60)

    for i, chunk in enumerate(chunks):

        print(f"\n{'=' * 60}")
        print(f"CHUNK {i + 1}")
        print(f"{'=' * 60}")

        print(
            f"Source : "
            f"{chunk.metadata.get('filename')}"
        )

        print(
            f"Taille : "
            f"{len(chunk.page_content)} caractères"
        )

        print(
            f"Index : "
            f"{chunk.metadata.get('chunk_index')}"
        )

        print("\nContenu :")

        print(chunk.page_content)


if __name__ == "__main__":
    main()