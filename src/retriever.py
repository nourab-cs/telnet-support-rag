class Retriever:
    """
    Gère la recherche sémantique.
    """

    def __init__(
        self,
        vectordb,
        k: int = 4,
    ):
        self.retriever = vectordb.as_retriever(
            search_kwargs={
                "k": k
            }
        )

    def get_retriever(self):
        return self.retriever