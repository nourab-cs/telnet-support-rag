from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from typing import Optional


class Generator:
    """
    Gère le LLM et la chaîne RAG avec l'API moderne LCEL.
    """

    def __init__(
        self,
        retriever,
        model_name: str = "mistral",
        temperature: float = 0.1,  # Augmenté légèrement pour plus de flexibilité
    ):

        self.retriever = retriever

        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature,
        )

    def create_chain(self, history: Optional[str] = None):

        # Template avec support pour l'historique
        if history:
            template = """
Tu es un assistant technique pour TELNET SmartConnect.

INSTRUCTIONS STRICTES:
- Ta reponse doit etre CONCISE et DIRECTE (max 3-4 phrases)
- Utilise UNIQUEMENT les informations du contexte fourni
- Sois factuel et precise
- Si l'information n'est pas presente, dis: "Je ne trouve pas cette information dans la documentation."
- Ne jamais inventer d'informations
- Tiens compte de l'historique de conversation pour comprendre le contexte

Historique de conversation:
{history}

Contexte:
{context}

Question:
{question}

Reponse concise et directe en français:
"""
        else:
            template = """
Tu es un assistant technique pour TELNET SmartConnect.

INSTRUCTIONS STRICTES:
- Ta reponse doit etre CONCISE et DIRECTE (max 3-4 phrases)
- Utilise UNIQUEMENT les informations du contexte fourni
- Sois factuel et precise
- Si l'information n'est pas presente, dis: "Je ne trouve pas cette information dans la documentation."
- Ne jamais inventer d'informations

Contexte:
{context}

Question:
{question}

Reponse concise et directe en français:
"""

        prompt = ChatPromptTemplate.from_template(template)

        # Créer la chaîne avec l'API LCEL moderne
        if history:
            self.chain = (
                {
                    "context": self.retriever | self.format_docs,
                    "question": RunnablePassthrough(),
                    "history": lambda x: history
                }
                | prompt
                | self.llm
                | StrOutputParser()
            )
        else:
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
    
    def invoke(self, question: str, history: Optional[str] = None):
        """Invoke the chain and return both result and sources"""
        # Recreate chain with history if provided
        if history and history.strip():
            self.create_chain(history)
        
        # Get sources first - use invoke instead of get_relevant_documents
        sources = self.retriever.invoke(question)
        
        # Get the answer
        result = self.chain.invoke(question)
        
        return {
            "result": result,
            "source_documents": sources
        }