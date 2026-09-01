from langchain_community.embeddings import HuggingFaceEmbeddings


class Embedder:
    """
    Gère le chargement du modèle d'embedding.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
        normalize_embeddings: bool = True,
    ):

        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings

        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={
                "device": self.device
            },
            encode_kwargs={
                "normalize_embeddings": self.normalize_embeddings
            },
        )

    def get_embeddings(self):
        """
        Retourne le modèle d'embedding.
        """

        return self.embedding_model