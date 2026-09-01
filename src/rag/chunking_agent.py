"""
Agent IA de chunking intelligent.

Cet agent utilise un LLM pour analyser le contenu des documents
et adapter dynamiquement les paramètres de chunking pour RAG.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import re


class ChunkingAgent:
    """
    Agent IA pour le chunking intelligent de documents optimisé pour RAG.
    """

    def __init__(self, model_name: str = "mistral"):
        """
        Initialise l'agent de chunking.

        Args:
            model_name: Nom du modele LLM a utiliser (default: mistral)
        """
        self.llm = ChatOllama(model=model_name, temperature=0.1)
        self.parser = JsonOutputParser()

        # Prompt pour l'analyse de chunking - optimisé pour RAG
        self.chunking_prompt = ChatPromptTemplate.from_messages([
            ("system", """Tu es un expert en traitement de documents techniques pour RAG.
Analyse le texte suivant et determine les meilleurs parametres de chunking.

Pour un système RAG optimal:
- Les chunks doivent être assez grands pour contenir du contexte sémantique complet
- Mais assez petits pour être précis lors de la recherche
- L'overlap doit assurer la continuité du contexte

Retourne ta reponse au format JSON avec:
- chunk_size: taille optimale des chunks (entre 400 et 1200 caracteres pour RAG)
- chunk_overlap: chevauchement optimal (entre 100 et 200 pour RAG)
- strategy: type de strategie a utiliser ("semantic", "structural", "standard")
- reason: breve explication du choix"""),
            ("user", "Contenu du document:\n\n{content}")
        ])

        self.chain = self.chunking_prompt | self.llm | self.parser

    def analyze_document(self, content: str) -> dict:
        """
        Analyse le contenu d'un document pour determiner lesParametres de chunking.

        Args:
            content: Contenu du document a analyser

        Returns:
            Dictionnaire avec les parametres de chunking
        """
        # Limiter la taille du contenu pour l'analyse (premiers 2000 caracteres)
        sample_content = content[:2000]

        try:
            result = self.chain.invoke({"content": sample_content})
            return {
                "chunk_size": min(max(result.get("chunk_size", 600), 400), 1200),
                "chunk_overlap": min(max(result.get("chunk_overlap", 120), 100), 200),
                "strategy": result.get("strategy", "standard"),
                "reason": result.get("reason", "Analyse standard")
            }
        except Exception as e:
            print(f"Erreur lors de l'analyse: {e}")
            # Retourner des valeurs par defaut optimisees pour RAG
            return {
                "chunk_size": 600,
                "chunk_overlap": 120,
                "strategy": "standard",
                "reason": "Erreur lors de l'analyse, valeurs par defaut RAG"
            }

    def detect_structure(self, content: str) -> dict:
        """
        Detecte la structure du document (titres, listes, sections).

        Args:
            content: Contenu du document

        Returns:
            Dictionnaire avec les caracteristiques de structure
        """
        structure = {
            "has_headers": bool(re.search(r'^#{1,3}\s', content, re.MULTILINE)),
            "has_lists": bool(re.search(r'^[\s]*[-*+]\s', content, re.MULTILINE)),
            "has_numbers": bool(re.search(r'^[\s]*\d+\.', content, re.MULTILINE)),
            "avg_line_length": len(content.split('\n')) / max(len(content.split('\n')), 1),
            "total_length": len(content)
        }
        return structure

    def get_adaptive_params(self, document: Document) -> dict:
        """
        Determine les parametres de chunking adaptes au document.

        Args:
            document: Document LangChain a analyser

        Returns:
            Dictionnaire avec les parametres de chunking
        """
        content = document.page_content

        # Detection de structure
        structure = self.detect_structure(content)

        # Heuristiques basees sur la structure - optimisees pour RAG
        if structure["has_headers"]:
            # Document avec structure - chunks moyens pour preserver le contexte
            base_size = 600
            base_overlap = 120
        elif structure["has_lists"] or structure["has_numbers"]:
            # Document avec listes - chunks moyens
            base_size = 650
            base_overlap = 130
        else:
            # Document texte pur - chunks plus grands pour le contexte
            base_size = 700
            base_overlap = 150

        # Ajustement selon la longueur totale
        if structure["total_length"] < 1000:
            base_size = min(base_size, 500)
            base_overlap = min(base_overlap, 100)
        elif structure["total_length"] > 10000:
            base_size = max(base_size, 800)
            base_overlap = max(base_overlap, 180)

        # Analyse LLM pour affiner les parametres
        llm_params = self.analyze_document(content)

        # Combinaison des approches (ponderation)
        final_chunk_size = int((base_size * 0.6) + (llm_params["chunk_size"] * 0.4))
        final_chunk_overlap = int((base_overlap * 0.6) + (llm_params["chunk_overlap"] * 0.4))

        return {
            "chunk_size": final_chunk_size,
            "chunk_overlap": final_chunk_overlap,
            "strategy": llm_params["strategy"],
            "reason": llm_params["reason"],
            "structure": structure
        }

    def chunk_document(self, document: Document) -> list[Document]:
        """
        Applique le chunking intelligent a un document.

        Args:
            document: Document LangChain a chunker

        Returns:
            Liste de documents chunkes
        """
        # Obtenir les parametres adaptes
        params = self.get_adaptive_params(document)

        print(f"Chunking pour document: {document.metadata.get('source', 'inconnu')}")
        print(f"  -> Chunk size: {params['chunk_size']}")
        print(f"  -> Overlap: {params['chunk_overlap']}")
        print(f"  -> Strategie: {params['strategy']}")
        print(f"  -> Raison: {params['reason']}")

        # Creer le splitter avec les parametres adaptes
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=params["chunk_size"],
            chunk_overlap=params["chunk_overlap"],
            separators=["\n\n", "\n", ". ", ", ", " ", ""]
        )

        # Appliquer le chunking
        chunks = splitter.split_documents([document])

        # Ajouter les metadonnees de chunking
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_index": i,
                "chunk_size": params["chunk_size"],
                "chunk_overlap": params["chunk_overlap"],
                "chunking_strategy": params["strategy"]
            })

        return chunks

    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        """
        Applique le chunking intelligent a une liste de documents.

        Args:
            documents: Liste de documents LangChain a chunker

        Returns:
            Liste de tous les documents chunkes
        """
        all_chunks = []

        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)

        return all_chunks
