from src.rag.document_loader import DocumentLoader
from src.rag.chunking_agent import ChunkingAgent


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
    # 2. Chunking intelligent avec agent IA
    # --------------------------------------------------

    chunking_agent = ChunkingAgent()

    chunks = chunking_agent.chunk_documents(documents)

    # --------------------------------------------------
    # 3. Afficher les résultats
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("RÉSULTATS DU CHUNKING AVEC AGENT IA")
    print("=" * 60)
    print(f"Total de chunks générés : {len(chunks)}")

    # Afficher les 10 premiers chunks en détail
    print("\n" + "=" * 60)
    print("CONTENU DES CHUNKS (10 premiers)")
    print("=" * 60)

    for i, chunk in enumerate(chunks[:10], 1):
        print(f"\n{'=' * 60}")
        print(f"CHUNK {i}")
        print(f"{'=' * 60}")

        source = chunk.metadata.get('source', chunk.metadata.get('filename', 'inconnu'))
        print(f"Source : {source}")
        print(f"Taille : {len(chunk.page_content)} caractères")
        print(f"Index : {chunk.metadata.get('chunk_index', 'N/A')}")
        print(f"Stratégie : {chunk.metadata.get('chunking_strategy', 'N/A')}")
        print(f"Chunk size : {chunk.metadata.get('chunk_size', 'N/A')}")
        print(f"Overlap : {chunk.metadata.get('chunk_overlap', 'N/A')}")

        print("\nContenu :")
        # Afficher le contenu en gérant les caractères Unicode
        try:
            content = chunk.page_content
            # Limiter l'affichage pour la lisibilité
            if len(content) > 350:
                content = content[:350] + "..."
            # Remplacer les emojis problématiques pour Windows console
            content = content.replace('✅', '[OK]').replace('❌', '[ERREUR]').replace('⚠️', '[ATTENTION]')
            print(content)
        except UnicodeEncodeError:
            # Fallback pour Windows console
            content = chunk.page_content.encode('utf-8', errors='ignore').decode('utf-8')
            if len(content) > 350:
                content = content[:350] + "..."
            print(content)

    # Statistiques
    print("\n" + "=" * 60)
    print("STATISTIQUES")
    print("=" * 60)

    chunk_sizes = [len(chunk.page_content) for chunk in chunks]
    print(f"Nombre total de chunks : {len(chunks)}")
    print(f"Taille moyenne : {sum(chunk_sizes) / len(chunk_sizes):.0f} caractères")
    print(f"Taille minimale : {min(chunk_sizes)} caractères")
    print(f"Taille maximale : {max(chunk_sizes)} caractères")

    # Compter les chunks par document source
    sources = {}
    for chunk in chunks:
        source = chunk.metadata.get('source', chunk.metadata.get('filename', 'inconnu'))
        sources[source] = sources.get(source, 0) + 1

    print("\nChunks par document source :")
    for source, count in sources.items():
        print(f"  - {source} : {count} chunks")


if __name__ == "__main__":
    main()
