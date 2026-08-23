from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


class Generator:
    """
    Gère le LLM et la chaîne RAG avec l'API moderne LCEL.
    """

    def __init__(
        self,
        retriever,
        model_name: str = "mistral",
        temperature: float = 0.1,
    ):

        self.retriever = retriever

        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature,
        )

    def create_chain(self):

        template = """
Tu es un assistant technique pour TELNET SmartConnect.

Tu réponds uniquement à partir du contexte fourni.

Si la réponse n'est pas présente dans le contexte, réponds :

"Je ne trouve pas cette information dans la documentation."

Contexte :
{context}

Question :
{question}

Réponse détaillée en français :
"""

        prompt = ChatPromptTemplate.from_template(template)

        # Créer la chaîne avec l'API LCEL moderne
        self.chain = (
            {
                "context": self.retriever | self.format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return self
    
    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def invoke(self, question: str):
        """Invoke the chain and return both result and sources"""
        # Get sources first - use invoke instead of get_relevant_documents
        sources = self.retriever.invoke(question)
        
        # Get the answer
        result = self.chain.invoke(question)
        
        return {
            "result": result,
            "source_documents": sources
        }