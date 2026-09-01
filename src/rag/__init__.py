"""
Package RAG (Retrieval-Augmented Generation) pour TELNET Support Bot.

Ce package contient tous les modules nécessaires au pipeline RAG:
- DocumentLoader: Chargement de documents
- Chunker: Chunking 
- Embedder: Génération d'embeddings
- ChromaStore: Base vectorielle ChromaDB
- Retriever: Récupération sémantique
- Generator: Génération de réponses
- Pipeline: Pipeline RAG complet
- ConversationHistory: Gestion de l'historique conversationnel
"""

from .document_loader import DocumentLoader
from .semantic_chunker import SemanticChunking
from .embedder import Embedder
from .chroma_store import ChromaStore
from .retriever import Retriever
from .generator import Generator
from .pipeline import RAGPipeline
from .conversation_history import ConversationHistory

__all__ = [
    "DocumentLoader",
    "SemanticChunking",
    "Embedder",
    "ChromaStore",
    "Retriever",
    "Generator",
    "RAGPipeline",
    "ConversationHistory",
]
