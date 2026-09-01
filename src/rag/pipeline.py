from .document_loader import DocumentLoader
from .semantic_chunker import SemanticChunking
from .embedder import Embedder
from .chroma_store import ChromaStore
from .retriever import Retriever
from .generator import Generator
from .conversation_history import ConversationHistory


class RAGPipeline:

    def __init__(

        self,

        data_dir,

        db_dir,

        collection_name="telnet_support",
        use_history: bool = True,
        max_history: int = 10

    ):

        self.data_dir = data_dir

        self.db_dir = db_dir

        self.collection_name = collection_name

        self.qa_chain = None
        
        # Historique de conversation
        self.use_history = use_history
        self.history = ConversationHistory(max_history=max_history) if use_history else None

    def build(self):

        print("=" * 70)
        print("Construction du pipeline RAG")
        print("=" * 70)

        # 1 Loader (inclut déjà un nettoyage léger)
        loader = DocumentLoader(self.data_dir)
        documents = loader.load()

        # 2 Chunking intelligent avec agent IA
        print("\nInitialisation de l'agent de chunking...")
        chunking_agent = SemanticChunking()
        chunks = chunking_agent.chunk_documents(documents)

        # 3 Embeddings
        embedder = Embedder()
        embeddings = embedder.get_embeddings()

        # 4 Chroma
        store = ChromaStore(
            persist_directory=self.db_dir,
            collection_name=self.collection_name,
        )

        vectordb = store.create(
            chunks,
            embeddings
        )

        # 5 Retriever - avec paramètres optimisés pour RAG
        retriever = Retriever(vectordb, k=4, search_type="similarity").get_retriever()

        # 6 Generator
        self.generator = Generator(retriever)
        self.generator.create_chain()

        print("\nPipeline prêt.\n")

    def ask(self, question: str):

        # Ajouter la question à l'historique
        if self.history:
            self.history.add_user_message(question)
        
        # Récupérer l'historique formaté si activé
        history_text = None
        if self.history and self.use_history:
            history_text = self.history.get_context_for_prompt()
        
        # Invoquer le générateur avec l'historique
        result = self.generator.invoke(question, history=history_text)
        
        # Ajouter la réponse à l'historique
        if self.history:
            sources = [doc.metadata.get('source', 'inconnu') for doc in result.get('source_documents', [])]
            self.history.add_assistant_message(result['result'], sources)
        
        return result
    
    def get_history(self):
        """Récupère l'historique de conversation."""
        if self.history:
            return self.history.get_formatted_history()
        return None
    
    def clear_history(self):
        """Efface l'historique de conversation."""
        if self.history:
            self.history.clear()
    
    def get_history_summary(self):
        """Récupère un résumé de la conversation."""
        if self.history:
            return self.history.get_conversation_summary()
        return "Historique non activé"