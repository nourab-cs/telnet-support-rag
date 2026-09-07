"""
Package RAG (Retrieval-Augmented Generation) pour TELNET Support Bot.

Ce package contient tous les modules nécessaires au pipeline RAG:
|- DocumentLoader: Chargement de documents
|- ChunkingAgent: Chunking intelligent
|- Embedder: Génération d'embeddings
|- ChromaStore: Base vectorielle ChromaDB
|- Retriever: Récupération sémantique
|- Generator: Génération de réponses
|- RAGPipeline: Pipeline RAG complet
|- ConversationHistory: Gestion de l'historique conversationnel
"""

from .document_loader import DocumentLoader
from .chunking_agent import ChunkingAgent
from .embedder import Embedder
from .chroma_store import ChromaStore
from .retriever import Retriever
from .generator import Generator
from .pipeline import RAGPipeline
from .conversation_history import ConversationHistory

__all__ = [
    "DocumentLoader",
    "ChunkingAgent",
    "Embedder",
    "ChromaStore",
    "Retriever",
    "Generator",
    "RAGPipeline",
    "ConversationHistory",
]