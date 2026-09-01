from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from pathlib import Path
import sys
import re

# Ajouter le dossier parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.document_loader import DocumentLoader
from src.text_cleaner import TextCleaner


# ============================================================
# 1. INITIALISATION DU LLM
# ============================================================

llm = ChatOllama(
    model="mistral",
    temperature=0
)


# ============================================================
# 2. CHARGEMENT DES DOCUMENTS
# ============================================================

print("Chargement des documents depuis ./data/...")

loader = DocumentLoader("./data")
documents = loader.load()

print(f"Documents chargés : {len(documents)}")


# ============================================================
# 3. NETTOYAGE
# ============================================================

print("Nettoyage des documents...")

cleaner = TextCleaner()
documents = cleaner.clean(documents)

print(f"Documents après nettoyage : {len(documents)}")


# ============================================================
# 4. FONCTION DE CHUNKING
# ============================================================

MAX_CHUNK_SIZE = 200


def agentic_chunk(text, max_size=MAX_CHUNK_SIZE):

    prompt = f"""
You are an expert document chunking agent.

Your task is to divide the text into small semantic chunks.

Rules:

1. Each chunk MUST be no longer than {max_size} characters.
2. Never split in the middle of a sentence if possible.
3. Keep sentences about the same topic together.
4. Preserve section titles with their content.
5. Do not rewrite the text.
6. Do not summarize the text.
7. Do not remove information.
8. Put <<<SPLIT>>> between chunks.
9. Return ONLY the original text with split markers.

Text:

{text}
"""

    response = llm.invoke(prompt)

    return response.content


# ============================================================
# 5. VALIDATION DES CHUNKS
# ============================================================

def validate_chunks(marked_text, max_size=MAX_CHUNK_SIZE):

    chunks = marked_text.split("<<<SPLIT>>>")

    valid_chunks = []
    oversized_chunks = []

    for chunk in chunks:

        chunk = chunk.strip()

        if not chunk:
            continue

        if len(chunk) <= max_size:
            valid_chunks.append(chunk)

        else:
            oversized_chunks.append(chunk)

    return valid_chunks, oversized_chunks


# ============================================================
# 6. FALLBACK : SPLIT PYTHON
# ============================================================

def split_large_chunk(text, max_size=MAX_CHUNK_SIZE):

    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current = ""

    for sentence in sentences:

        if not current:
            current = sentence

        elif len(current) + 1 + len(sentence) <= max_size:
            current += " " + sentence

        else:
            chunks.append(current.strip())
            current = sentence

    if current:
        chunks.append(current.strip())

    return chunks


# ============================================================
# 7. TRAITEMENT DES DOCUMENTS
# ============================================================

all_chunks = []


for doc_index, doc in enumerate(documents):

    filename = doc.metadata.get(
        "filename",
        f"document_{doc_index}"
    )

    print(f"\nTraitement : {filename}")

    text = doc.page_content

    # Appel de l'agent IA
    marked_text = agentic_chunk(
        text,
        MAX_CHUNK_SIZE
    )

    # Validation
    valid_chunks, oversized_chunks = validate_chunks(
        marked_text,
        MAX_CHUNK_SIZE
    )

    # Ajouter les chunks valides
    for chunk in valid_chunks:

        all_chunks.append(
            Document(
                page_content=chunk,
                metadata={
                    **doc.metadata,
                    "chunk_size": len(chunk),
                    "chunking_method": "agentic"
                }
            )
        )

    # Corriger les chunks trop grands
    for oversized in oversized_chunks:

        print(
            f"Chunk trop grand détecté : "
            f"{len(oversized)} caractères"
        )

        corrected_chunks = split_large_chunk(
            oversized,
            MAX_CHUNK_SIZE
        )

        for chunk in corrected_chunks:

            all_chunks.append(
                Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_size": len(chunk),
                        "chunking_method": "agentic_fallback"
                    }
                )
            )


# ============================================================
# 8. AFFICHAGE
# ============================================================

print("\n")
print("=" * 60)
print("AGENTIC CHUNKING RESULTS")
print("=" * 60)

print(f"Nombre total de chunks : {len(all_chunks)}")

for i, chunk in enumerate(all_chunks, 1):

    print(f"\nChunk {i}")
    print(f"Taille : {len(chunk.page_content)} caractères")
    print(f"Source : {chunk.metadata.get('filename')}")
    print("-" * 40)
    print(chunk.page_content)