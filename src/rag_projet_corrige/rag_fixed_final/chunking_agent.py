from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_experimental.text_splitter import SemanticChunker


# ============================================================
# METRICS
# ============================================================

@dataclass
class ChunkingMetrics:
    avg_chunk_size: float
    min_chunk_size: int
    max_chunk_size: int
    total_chunks: int
    avg_overlap: float


# ============================================================
# CHUNKING AGENT
# ============================================================

class ChunkingAgent:
    """
    Agent IA de chunking adaptatif pour un système RAG.

    Le LLM analyse le document et choisit une stratégie :

    - structural : structure Markdown / sections
    - semantic   : rupture de sens
    - recursive  : fallback classique

    Le LLM ne produit PAS les chunks.
    Il décide uniquement comment ils doivent être produits.
    """

    def __init__(
        self,
        model_name: str = "mistral",
        embeddings=None,
        enable_llm_analysis: bool = True,
        target_chunk_size: int = 900,
        min_chunk_size: int = 300,
        max_chunk_size: int = 1200,
        cache_dir: str = "./chunking_cache",
    ):
        """
        Args:
            model_name:
                Modèle Ollama utilisé par l'agent.

            embeddings:
                Modèle d'embeddings utilisé par SemanticChunker.

            enable_llm_analysis:
                Active ou non l'agent IA.

            target_chunk_size:
                Taille cible approximative.
                Ce n'est PAS une limite stricte.

            min_chunk_size:
                Taille minimale acceptable.

            max_chunk_size:
                Taille maximale de sécurité.

            cache_dir:
                Répertoire du cache.
        """

        self.model_name = model_name
        self.embeddings = embeddings
        self.enable_llm_analysis = enable_llm_analysis

        self.target_chunk_size = target_chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # LLM
        # ----------------------------------------------------

        self.llm = None
        self.parser = None
        self.chain = None

        if self.enable_llm_analysis:

            self.llm = ChatOllama(
                model=model_name,
                temperature=0,
            )

            self.parser = JsonOutputParser()

            self.chunking_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """
Tu es un agent expert en segmentation de documents
pour un système RAG.

Ta mission est d'analyser le document et de choisir
la meilleure stratégie de chunking.

IMPORTANT :

Tu ne dois PAS produire les chunks.

Tu dois uniquement prendre une décision.

Stratégies disponibles :

1. structural

Utiliser cette stratégie si le document possède
une structure claire :

- titres
- sous-titres
- sections
- listes
- tableaux
- documentation technique

2. semantic

Utiliser cette stratégie si le document est peu structuré
mais contient plusieurs sujets ou changements de sens.

L'objectif est que chaque chunk contienne une idée ou
un sujet cohérent et que la fin du chunk corresponde
autant que possible à une fin sémantique.

3. recursive

Utiliser cette stratégie pour les documents simples
ou lorsque les deux autres stratégies ne sont pas adaptées.

La taille cible est d'environ 900 caractères.

IMPORTANT :

900 caractères est une TAILLE CIBLE,
pas une frontière obligatoire.

La cohérence sémantique est prioritaire
sur la taille exacte.

Contraintes :

- target_chunk_size : environ 900
- min_chunk_size : 300
- max_chunk_size : 1200

Retourne UNIQUEMENT un JSON valide sous la forme :

{
    "document_type": "...",
    "strategy": "structural | semantic | recursive",
    "target_chunk_size": 900,
    "min_chunk_size": 300,
    "max_chunk_size": 1200,
    "reason": "..."
}

Document à analyser :
""",
                    ),
                    (
                        "user",
                        "{content}",
                    ),
                ]
            )

            self.chain = (
                self.chunking_prompt
                | self.llm
                | self.parser
            )

        # ----------------------------------------------------
        # CACHE
        # ----------------------------------------------------

        self._analysis_cache: Dict[str, Dict[str, Any]] = {}

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        self.metrics: Optional[ChunkingMetrics] = None

    # ========================================================
    # ANALYSE DU DOCUMENT
    # ========================================================

    def analyze_document(
        self,
        content: str,
    ) -> Dict[str, Any]:
        """
        L'agent IA analyse le document
        et choisit une stratégie.
        """

        if not content or not content.strip():
            return self._default_parameters()

        # ----------------------------------------------------
        # Hash stable
        # ----------------------------------------------------

        content_hash = hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()

        # ----------------------------------------------------
        # Vérification du cache
        # ----------------------------------------------------

        if content_hash in self._analysis_cache:

            return self._analysis_cache[content_hash]

        # ----------------------------------------------------
        # Analyse LLM
        # ----------------------------------------------------

        if self.enable_llm_analysis and self.chain:

            try:

                # Éviter d'envoyer un document énorme au LLM
                sample_content = content[:6000]

                result = self.chain.invoke(
                    {
                        "content": sample_content
                    }
                )

                params = self._validate_agent_result(
                    result
                )

                self._analysis_cache[
                    content_hash
                ] = params

                return params

            except Exception as e:
                pass

        # ----------------------------------------------------
        # FALLBACK HEURISTIQUE
        # ----------------------------------------------------

        params = self._heuristic_analysis(
            content
        )

        self._analysis_cache[
            content_hash
        ] = params

        return params

    # ========================================================
    # VALIDATION DE LA DECISION DU LLM
    # ========================================================

    def _validate_agent_result(
        self,
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Valide et nettoie la réponse du LLM.
        """

        strategy = str(
            result.get(
                "strategy",
                "recursive",
            )
        ).lower().strip()

        allowed_strategies = {
            "structural",
            "semantic",
            "recursive",
        }

        if strategy not in allowed_strategies:
            strategy = "recursive"

        target = self._safe_int(
            result.get(
                "target_chunk_size",
                self.target_chunk_size,
            ),
            self.target_chunk_size,
        )

        minimum = self._safe_int(
            result.get(
                "min_chunk_size",
                self.min_chunk_size,
            ),
            self.min_chunk_size,
        )

        maximum = self._safe_int(
            result.get(
                "max_chunk_size",
                self.max_chunk_size,
            ),
            self.max_chunk_size,
        )

        # ----------------------------------------------------
        # Sécurisation des valeurs
        # ----------------------------------------------------

        target = max(
            400,
            min(target, 1200),
        )

        minimum = max(
            100,
            min(minimum, target),
        )

        maximum = max(
            target,
            min(maximum, 1600),
        )

        return {
            "document_type": result.get(
                "document_type",
                "unknown",
            ),

            "strategy": strategy,

            "target_chunk_size": target,

            "min_chunk_size": minimum,

            "max_chunk_size": maximum,

            "reason": result.get(
                "reason",
                "Décision automatique.",
            ),
        }

    # ========================================================
    # ANALYSE HEURISTIQUE
    # ========================================================

    def _heuristic_analysis(
        self,
        content: str,
    ) -> Dict[str, Any]:
        """
        Fallback lorsque le LLM n'est pas disponible.
        """

        structure = self.detect_structure(
            content
        )

        if structure["has_headers"]:

            strategy = "structural"

            reason = (
                "Le document possède une structure "
                "Markdown avec des titres."
            )

        elif structure["paragraph_count"] >= 5:

            strategy = "semantic"

            reason = (
                "Le document contient plusieurs paragraphes "
                "et nécessite une segmentation sémantique."
            )

        else:

            strategy = "recursive"

            reason = (
                "Document simple : utilisation du "
                "RecursiveCharacterTextSplitter."
            )

        return {
            "document_type": "technical_document",

            "strategy": strategy,

            "target_chunk_size": (
                self.target_chunk_size
            ),

            "min_chunk_size": (
                self.min_chunk_size
            ),

            "max_chunk_size": (
                self.max_chunk_size
            ),

            "reason": reason,
        }

    # ========================================================
    # DETECTION DE STRUCTURE
    # ========================================================

    def detect_structure(
        self,
        content: str,
    ) -> Dict[str, Any]:
        """
        Détecte les différentes structures
        présentes dans le document.
        """

        lines = content.splitlines()

        return {

            # Titres Markdown
            "has_headers": bool(
                re.search(
                    r"^#{1,6}\s+",
                    content,
                    re.MULTILINE,
                )
            ),

            # Listes à puces
            "has_lists": bool(
                re.search(
                    r"^\s*[-*+]\s+",
                    content,
                    re.MULTILINE,
                )
            ),

            # Listes numérotées
            "has_numbered_lists": bool(
                re.search(
                    r"^\s*\d+[.)]\s+",
                    content,
                    re.MULTILINE,
                )
            ),

            # Blocs de code Markdown
            "has_code_blocks": "```" in content,

            # Tableaux Markdown
            "has_tables": bool(
                re.search(
                    r"^\s*\|.*\|.*$",
                    content,
                    re.MULTILINE,
                )
            ),

            # Citations Markdown
            "has_quotes": bool(
                re.search(
                    r"^\s*>\s+",
                    content,
                    re.MULTILINE,
                )
            ),

            # Longueur totale
            "total_length": len(content),

            # Nombre de lignes
            "line_count": len(lines),

            # Nombre de blocs séparés par une ligne vide.
            "paragraph_count": len(
                [
                    block
                    for block in re.split(r"\n\s*\n", content.strip())
                    if block.strip()
                ]
            ),

            # Taille moyenne des lignes
            "avg_line_length": (
                sum(
                    len(line)
                    for line in lines
                )
                / max(len(lines), 1)
            ),
        }

    # ========================================================
    # PARAMETRES ADAPTATIFS
    # ========================================================

    def get_adaptive_params(
        self,
        document: Document,
    ) -> Dict[str, Any]:

        content = document.page_content

        params = self.analyze_document(
            content
        )

        return {
            **params,
            "structure": self.detect_structure(
                content
            ),
        }

    # ========================================================
    # CHUNKING PRINCIPAL
    # ========================================================

    def chunk_document(
        self,
        document: Document,
    ) -> List[Document]:

        params = self.get_adaptive_params(
            document
        )

        strategy = params["strategy"]

        # ----------------------------------------------------
        # Sélection du splitter
        # ----------------------------------------------------

        if strategy == "structural":

            chunks = self._structural_chunking(
                document,
                params,
            )

        elif strategy == "semantic":

            chunks = self._semantic_chunking(
                document,
                params,
            )

        else:

            chunks = self._recursive_chunking(
                document,
                params,
            )

        # ----------------------------------------------------
        # Validation / métadonnées
        # ----------------------------------------------------

        chunks = self._post_process_chunks(
            chunks,
            document,
            params,
        )

        return chunks

    # ========================================================
    # STRUCTURAL CHUNKING
    # ========================================================

    def _structural_chunking(
        self,
        document: Document,
        params: Dict[str, Any],
    ) -> List[Document]:
        """
        Découpe selon les titres Markdown.

        Exemple :

        # Device

        ## Configuration

        ...

        ## Diagnostic

        ...

        Chaque section conserve son contexte.
        """

        headers_to_split_on = [
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
            ("####", "h4"),
        ]

        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,
        )

        chunks = splitter.split_text(
            document.page_content
        )

        result = []

        for chunk in chunks:

            metadata = dict(
                document.metadata
            )

            metadata.update(
                chunk.metadata
            )

            result.append(
                Document(
                    page_content=chunk.page_content,
                    metadata=metadata,
                )
            )

        # ----------------------------------------------------
        # Si une section est trop grande,
        # on la subdivise.
        # ----------------------------------------------------

        final_chunks = []

        for chunk in result:

            if (
                len(chunk.page_content)
                <= params["max_chunk_size"]
            ):

                final_chunks.append(
                    chunk
                )

            else:

                sub_chunks = (
                    self._semantic_or_recursive(
                        chunk,
                        params,
                    )
                )

                final_chunks.extend(
                    sub_chunks
                )

        return final_chunks

    # ========================================================
    # SEMANTIC CHUNKING
    # ========================================================

    def _semantic_chunking(
        self,
        document: Document,
        params: Dict[str, Any],
    ) -> List[Document]:
        """
        Découpage basé sur les changements sémantiques.
        """

        if self.embeddings is None:

            return self._recursive_chunking(
                document,
                params,
            )

        try:

            splitter = SemanticChunker(
                self.embeddings,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=85,
                min_chunk_size=params[
                    "min_chunk_size"
                ],
            )

            chunks = splitter.split_documents(
                [document]
            )

            return chunks

        except Exception as e:

            return self._recursive_chunking(
                document,
                params,
            )

    # ========================================================
    # RECURSIVE CHUNKING
    # ========================================================

    def _recursive_chunking(
        self,
        document: Document,
        params: Dict[str, Any],
    ) -> List[Document]:
        """
        Fallback robuste.
        """

        target = params[
            "target_chunk_size"
        ]

        # Overlap de sécurité
        overlap = min(
            150,
            target // 5,
        )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=target,
            chunk_overlap=overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                "? ",
                "! ",
                "; ",
                ", ",
                " ",
                "",
            ],
            length_function=len,
            keep_separator=True,
        )

        return splitter.split_documents(
            [document]
        )

    # ========================================================
    # SEMANTIC OU RECURSIVE
    # ========================================================

    def _semantic_or_recursive(
        self,
        document: Document,
        params: Dict[str, Any],
    ) -> List[Document]:

        if self.embeddings is not None:

            try:

                return self._semantic_chunking(
                    document,
                    params,
                )

            except Exception:
                pass

        return self._recursive_chunking(
            document,
            params,
        )

    # ========================================================
    # POST-PROCESSING
    # ========================================================

    def _post_process_chunks(
        self,
        chunks: List[Document],
        document: Document,
        params: Dict[str, Any],
    ) -> List[Document]:

        source = document.metadata.get(
            "source",
            document.metadata.get(
                "filename",
                "inconnu",
            ),
        )

        # ----------------------------------------------------
        # Filtrer les chunks trop petits
        # ----------------------------------------------------

        filtered = []

        for chunk in chunks:

            content = chunk.page_content.strip()

            if len(content) < params["min_chunk_size"]:
                # Fusionner seulement si cela reste dans la limite maximale.
                if filtered:
                    previous = filtered[-1]
                    merged = (
                        previous.page_content.rstrip()
                        + "\n\n"
                        + content
                    )
                    if len(merged) <= params["max_chunk_size"]:
                        previous.page_content = merged
                    else:
                        filtered.append(
                            Document(
                                page_content=content,
                                metadata=dict(chunk.metadata),
                            )
                        )
                else:
                    pass
                continue

            filtered.append(
                Document(
                    page_content=content,
                    metadata=dict(
                        chunk.metadata
                    ),
                )
            )

        # ----------------------------------------------------
        # Ajouter les métadonnées
        # ----------------------------------------------------

        total_chunks = len(filtered)

        for index, chunk in enumerate(
            filtered
        ):

            chunk.metadata.update(
                {
                    "source": source,

                    "chunk_index": index,

                    "chunk_id": (
                        f"{document.metadata.get('document_id', source)}:chunk:{index}"
                    ),

                    "total_chunks": total_chunks,

                    "chunking_strategy": params[
                        "strategy"
                    ],

                    "target_chunk_size": params[
                        "target_chunk_size"
                    ],

                    "min_chunk_size": params[
                        "min_chunk_size"
                    ],

                    "max_chunk_size": params[
                        "max_chunk_size"
                    ],

                    "chunk_length": len(
                        chunk.page_content
                    ),

                    "chunking_reason": params[
                        "reason"
                    ],
                }
            )

        return filtered

    # ========================================================
    # CHUNKING DE PLUSIEURS DOCUMENTS
    # ========================================================

    def chunk_documents(
        self,
        documents: List[Document],
    ) -> List[Document]:

        if not documents:
            return []

        cache_key = self._generate_cache_key(
            documents
        )

        cache_file = (
            self.cache_dir
            / f"chunks_{cache_key}.json"
        )

        # ----------------------------------------------------
        # Cache
        # ----------------------------------------------------

        if cache_file.exists():

            try:

                chunks = (
                    self._load_chunks_from_cache(
                        cache_file
                    )
                )

                self._calculate_metrics(
                    chunks
                )

                return chunks

            except Exception as e:

                cache_file.unlink(
                    missing_ok=True
                )

        # ----------------------------------------------------
        # Chunking
        # ----------------------------------------------------

        all_chunks = []

        for document in documents:

            chunks = self.chunk_document(
                document
            )

            all_chunks.extend(
                chunks
            )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        self._calculate_metrics(
            all_chunks
        )

        # ----------------------------------------------------
        # Save cache
        # ----------------------------------------------------

        self._save_chunks_to_cache(
            cache_file,
            all_chunks,
        )

        return all_chunks

    # ========================================================
    # METRICS
    # ========================================================

    def _calculate_metrics(
        self,
        chunks: List[Document],
    ) -> None:

        if not chunks:

            self.metrics = None

            return

        sizes = [
            len(chunk.page_content)
            for chunk in chunks
        ]

        self.metrics = ChunkingMetrics(
            avg_chunk_size=(
                sum(sizes) / len(sizes)
            ),

            min_chunk_size=min(sizes),

            max_chunk_size=max(sizes),

            total_chunks=len(sizes),

            # SemanticChunker ne fonctionne
            # pas avec un overlap fixe.
            avg_overlap=0.0,
        )

    # ========================================================
    # CACHE KEY
    # ========================================================

    def _generate_cache_key(
        self,
        documents: List[Document],
    ) -> str:

        hasher = hashlib.sha256()

        # Trier les documents pour avoir
        # un hash stable.
        for document in sorted(
            documents,
            key=lambda x: x.metadata.get(
                "source",
                "",
            ),
        ):

            hasher.update(
                document.page_content.encode(
                    "utf-8"
                )
            )

            hasher.update(
                json.dumps(
                    document.metadata,
                    sort_keys=True,
                    default=str,
                ).encode("utf-8")
            )

        # ----------------------------------------------------
        # Les paramètres du chunking
        # font également partie du hash.
        # ----------------------------------------------------

        hasher.update(
            str(
                self.target_chunk_size
            ).encode()
        )

        hasher.update(
            str(
                self.min_chunk_size
            ).encode()
        )

        hasher.update(
            str(
                self.max_chunk_size
            ).encode()
        )

        return hasher.hexdigest()[:16]

    # ========================================================
    # SAVE CACHE
    # ========================================================

    def _save_chunks_to_cache(
        self,
        cache_file: Path,
        chunks: List[Document],
    ) -> None:

        try:

            data = [
                {
                    "page_content": chunk.page_content,
                    "metadata": chunk.metadata,
                }
                for chunk in chunks
            ]

            with open(
                cache_file,
                "w",
                encoding="utf-8",
            ) as file:

                json.dump(
                    data,
                    file,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )

        except Exception as e:
            pass

    # ========================================================
    # LOAD CACHE
    # ========================================================

    def _load_chunks_from_cache(
        self,
        cache_file: Path,
    ) -> List[Document]:

        with open(
            cache_file,
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

        return [
            Document(
                page_content=item[
                    "page_content"
                ],
                metadata=item[
                    "metadata"
                ],
            )
            for item in data
        ]

    # ========================================================
    # DEFAULT PARAMETERS
    # ========================================================

    def _default_parameters(
        self,
    ) -> Dict[str, Any]:

        return {
            "document_type": "unknown",

            "strategy": "recursive",

            "target_chunk_size": (
                self.target_chunk_size
            ),

            "min_chunk_size": (
                self.min_chunk_size
            ),

            "max_chunk_size": (
                self.max_chunk_size
            ),

            "reason": (
                "Document vide ou analyse impossible."
            ),
        }

    # ========================================================
    # SAFE INTEGER
    # ========================================================

    @staticmethod
    def _safe_int(
        value: Any,
        default: int,
    ) -> int:

        try:

            return int(value)

        except (
            TypeError,
            ValueError,
        ):

            return default

    # ========================================================
    # GET METRICS
    # ========================================================

    def get_metrics(
        self,
    ) -> Optional[ChunkingMetrics]:

        return self.metrics