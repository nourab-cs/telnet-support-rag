from document_loader import DocumentLoader
from text_cleaner import TextCleaner
from chunker import Chunker
from embedder import Embedder
from chroma_store import ChromaStore
from retriever import Retriever
from generator import Generator


class RAGPipeline:

    def __init__(

        self,

        data_dir,

        db_dir,

        collection_name="telnet_support"

    ):

        self.data_dir = data_dir

        self.db_dir = db_dir

        self.collection_name = collection_name

        self.qa_chain = None

    def build(self):

        print("=" * 70)
        print("Construction du pipeline RAG")
        print("=" * 70)

        # 1 Loader
        loader = DocumentLoader(self.data_dir)
        documents = loader.load()

        # 2 Nettoyage
        cleaner = TextCleaner()
        documents = cleaner.clean(documents)

        # 3 Chunking
        chunker = Chunker()
        chunks = chunker.split(documents)

        # 4 Embeddings
        embedder = Embedder()
        embeddings = embedder.get_embeddings()

        # 5 Chroma
        store = ChromaStore(

            persist_directory=self.db_dir,

            collection_name=self.collection_name,

        )

        vectordb = store.create(

            chunks,

            embeddings

        )

        # 6 Retriever
        retriever = Retriever(vectordb).get_retriever()

        # 7 Generator
        self.generator = Generator(retriever)
        self.generator.create_chain()

        print("\nPipeline prêt.\n")

    def ask(self, question: str):

        result = self.generator.invoke(question)
        return result