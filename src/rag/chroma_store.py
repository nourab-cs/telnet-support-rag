from pathlib import Path

from langchain_chroma import Chroma


class ChromaStore:
    """
    Création et chargement de la base vectorielle Chroma.
    """

    def __init__(
        self,
        persist_directory: str,
        collection_name: str = "telnet_support",
    ):

        self.persist_directory = persist_directory
        self.collection_name = collection_name

    def create(self, chunks, embeddings):
        """
        Crée ou recharge une base Chroma.
        """

        db_path = Path(self.persist_directory)

        if db_path.exists() and any(db_path.iterdir()):

            print("Chargement de la base existante...")

            vectordb = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=embeddings,
                collection_name=self.collection_name,
            )

        else:

            print("Creation de la base vectorielle...")

            vectordb = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name,
            )

        return vectordb