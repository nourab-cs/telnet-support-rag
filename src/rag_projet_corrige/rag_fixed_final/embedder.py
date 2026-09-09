from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path
import os


class Embedder:
    """
    Gère le chargement du modèle d'embedding avec optimisations.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
        normalize_embeddings: bool = True,
        cache_folder: str = "./models_cache",
        **kwargs
    ):

        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.cache_folder = Path(cache_folder)
        self.cache_folder.mkdir(parents=True, exist_ok=True)

        # Configuration HF pour le cache (localisée à cette instance)
        self._setup_hf_cache()

        try:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={
                    "device": self.device,
                    "trust_remote_code": True
                },
                encode_kwargs={
                    "normalize_embeddings": self.normalize_embeddings,
                    "batch_size": 64,  # Optimisé pour CPU
                },
                cache_folder=str(self.cache_folder),
            )

        except Exception as e:
            raise

    def _setup_hf_cache(self):
        """Configure le cache Hugging Face de manière locale."""
        # Configuration HF pour le cache (uniquement pour cette instance)
        original_hf_home = os.environ.get('HF_HOME')
        original_transformers_cache = os.environ.get('TRANSFORMERS_CACHE')
        original_hf_datasets_cache = os.environ.get('HF_DATASETS_CACHE')

        try:
            os.environ['HF_HOME'] = str(self.cache_folder)
            os.environ['TRANSFORMERS_CACHE'] = str(self.cache_folder / 'transformers')
            os.environ['HF_DATASETS_CACHE'] = str(self.cache_folder / 'datasets')
        except Exception as e:
            # Restaurer les valeurs originales en cas d'erreur
            if original_hf_home:
                os.environ['HF_HOME'] = original_hf_home
            if original_transformers_cache:
                os.environ['TRANSFORMERS_CACHE'] = original_transformers_cache
            if original_hf_datasets_cache:
                os.environ['HF_DATASETS_CACHE'] = original_hf_datasets_cache

    def get_embeddings(self):
        """
        Retourne le modèle d'embedding.
        """
        return self.embedding_model
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed une liste de documents avec optimisation batch.
        
        Args:
            texts: Liste de textes à embedder
            
        Returns:
            Liste d'embeddings
        """
        if not texts:
            return []
        
        try:
            return self.embedding_model.embed_documents(texts)
        except Exception as e:
            raise
    
    def embed_query(self, text: str) -> list[float]:
        """
        Embed une requête.
        
        Args:
            text: Texte à embedder
            
        Returns:
            Embedding
        """
        try:
            return self.embedding_model.embed_query(text)
        except Exception as e:
            raise