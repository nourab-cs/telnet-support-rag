import re
from typing import List

from langchain_core.documents import Document


class TextCleaner:
    """
    Nettoie les documents avant le chunking.
    """

    def __init__(self, min_length: int = 30):
        self.min_length = min_length

    def clean(self, documents: List[Document]) -> List[Document]:
        """
        Nettoie une liste de documents.

        Args:
            documents: Liste des documents LangChain

        Returns:
            Liste des documents nettoyés
        """

        cleaned_documents = []

        for doc in documents:

            text = doc.page_content

            # Supprimer les espaces inutiles
            text = re.sub(r"\s+", " ", text)

            # Supprimer les lignes vides
            text = re.sub(r"\n+", "\n", text)

            # Supprimer les tabulations
            text = text.replace("\t", " ")

            # Enlever les espaces en début/fin
            text = text.strip()

            # Ignorer les documents trop courts
            if len(text) < self.min_length:
                continue

            doc.page_content = text

            cleaned_documents.append(doc)

        print(f"Documents apres nettoyage : {len(cleaned_documents)}")

        return cleaned_documents