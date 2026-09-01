class Retriever:
    """
    Gère la recherche sémantique optimisée pour RAG.
    """

    def __init__(
        self,
        vectordb,
        k: int = 4,  # Réduit à 4 pour une meilleure précision
        search_type: str = "similarity",  # similarity, mmr, similarity_score_threshold
    ):
        """
        Initialise le retriever avec des paramètres optimisés pour RAG.

        Args:
            vectordb: Base vectorielle ChromaDB
            k: Nombre de documents à récupérer (4 est optimal pour RAG)
            search_type: Type de recherche (similarity, mmr, similarity_score_threshold)
        """
        self.search_type = search_type
        self.k = k

        if search_type == "mmr":
            # MMR (Maximal Marginal Relevance) pour la diversité
            self.retriever = vectordb.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": k,
                    "fetch_k": k * 2,  # Récupère plus de documents pour la diversité
                    "lambda_mult": 0.5  # Balance entre similarité et diversité
                }
            )
        elif search_type == "similarity_score_threshold":
            # Filtrage par score de similarité
            self.retriever = vectordb.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": k,
                    "score_threshold": 0.5  # Seuil de similarité minimum
                }
            )
        else:
            # Similarité standard (par défaut)
            self.retriever = vectordb.as_retriever(
                search_kwargs={
                    "k": k
                }
            )

    def get_retriever(self):
        return self.retriever