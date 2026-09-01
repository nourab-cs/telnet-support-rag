"""
Chunking sémantique amélioré pour documents techniques.

Utilise une approche hybride : structurel + sémantique
pour mieux préserver la cohérence des documents techniques.
"""

from typing import List
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker as LangChainSemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re


class SemanticChunking:
    """
    Chunking sémantique amélioré pour documents techniques.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_amount: float = 95,  # Plus élevé pour moins de chunks plus grands
        use_hybrid: bool = True,  # Mode hybride structurel + sémantique
        min_chunk_size: int = 200,  # Taille minimale augmentée
        max_section_chunks: int = 5  # Limiter le nombre de chunks par section
    ):
        """
        Initialise le chunker sémantique amélioré.

        Args:
            model_name: Modèle d'embeddings (BGE-M3 par défaut)
            breakpoint_threshold_type: Type de seuil ("percentile", "standard_deviation", "gradient")
            breakpoint_threshold_amount: Valeur du seuil (95 pour des chunks plus grands)
            use_hybrid: Utiliser le mode hybride structurel + sémantique
            min_chunk_size: Taille minimale des chunks (200 caractères)
            max_section_chunks: Nombre maximum de chunks par section
        """
        self.use_hybrid = use_hybrid
        self.min_chunk_size = min_chunk_size
        self.max_section_chunks = max_section_chunks
        
        # Créer les embeddings avec BGE-M3
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Initialiser le SemanticChunker de LangChain
        self.semantic_splitter = LangChainSemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount,
        )

        # Splitter structurel de secours avec paramètres optimisés
        self.structure_splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,  # Plus grand pour de meilleurs chunks
            chunk_overlap=180,  # Overlap augmenté pour le contexte
            separators=["\n\n\n", "\n\n", "\n", "###", "##", "#", ". ", ", ", " ", ""]
        )

    def preprocess_document(self, text: str) -> str:
        """
        Pré-traite le document pour améliorer le chunking.

        Args:
            text: Texte du document

        Returns:
            Texte pré-traité
        """
        # Normaliser les sauts de ligne multiples
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # S'assurer que les titres sont bien formés
        text = re.sub(r'^(#{1,3})([^\s#])', r'\1 \2', text, flags=re.MULTILINE)
        
        return text

    def split_by_structure(self, text: str) -> List[str]:
        """
        Divise le texte selon la structure (titres, sections).

        Args:
            text: Texte à diviser

        Returns:
            Liste de sections
        """
        # Diviser par les titres principaux et sous-titres
        sections = re.split(r'\n(?=#{1,3}\s)', text)
        
        # Filtrer les sections vides et trop courtes
        sections = [s.strip() for s in sections if s.strip() and len(s.strip()) > 100]
        
        return sections if sections else [text]

    def chunk_document(self, document: Document) -> List[Document]:
        """
        Applique le chunking sémantique amélioré à un document.

        Args:
            document: Document LangChain

        Returns:
            Liste de documents chunkés
        """
        text = document.page_content
        
        # Pré-traitement
        text = self.preprocess_document(text)

        chunks = []

        if self.use_hybrid:
            # Mode hybride : d'abord structurel, puis sémantique dans chaque section
            sections = self.split_by_structure(text)
            
            for i, section in enumerate(sections):
                if len(section) < 200:  # Ignorer les sections trop courtes (augmenté)
                    continue
                    
                # Chunking sémantique de chaque section
                try:
                    section_chunks = self.semantic_splitter.split_text(section)
                except Exception as e:
                    # Fallback au chunking structurel
                    section_chunks = self.structure_splitter.split_text(section)
                
                # Limiter le nombre de chunks par section
                section_chunks = section_chunks[:self.max_section_chunks]
                
                for j, chunk_text in enumerate(section_chunks):
                    if len(chunk_text.strip()) < self.min_chunk_size:  # Utiliser le nouveau seuil minimum
                        continue
                        
                    chunks.append(
                        Document(
                            page_content=chunk_text.strip(),
                            metadata={
                                **document.metadata,
                                "chunk_index": len(chunks),
                                "chunking_method": "hybrid",
                                "chunk_size": len(chunk_text),
                                "section_index": i
                            }
                        )
                    )
        else:
            # Mode purement sémantique
            try:
                chunk_texts = self.semantic_splitter.split_text(text)
            except Exception as e:
                print(f"Erreur chunking sémantique: {e}, fallback au chunking structurel")
                chunk_texts = self.structure_splitter.split_text(text)

            for i, chunk_text in enumerate(chunk_texts):
                if len(chunk_text.strip()) < self.min_chunk_size:  # Utiliser le nouveau seuil minimum
                    continue
                    
                chunks.append(
                    Document(
                        page_content=chunk_text.strip(),
                        metadata={
                            **document.metadata,
                            "chunk_index": i,
                            "chunking_method": "semantic",
                            "chunk_size": len(chunk_text)
                        }
                    )
                )

        return chunks

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Applique le chunking sémantique amélioré à une liste de documents.

        Args:
            documents: Liste de documents LangChain

        Returns:
            Liste de tous les documents chunkés
        """
        all_chunks = []

        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)

        print(f"Chunking sémantique amélioré: {len(all_chunks)} chunks générés")

        return all_chunks
