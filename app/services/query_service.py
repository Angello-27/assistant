import openai
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from app.core.config import settings
from app.schemas.response import QueryResponse  # ⬅ importalo al inicio

# Configurar la API key de OpenAI
openai.api_key = settings.OPENAI_API_KEY


class QueryService:
    def __init__(self, document_retriever=None):
        """
        document_retriever: instancia de DocumentRetriever (opcional).
        Si se proporciona, se usará para la búsqueda en documentos locales.
        """
        self.document_retriever = document_retriever

    def query(self, query: str, use_retrieval: bool = True) -> QueryResponse:
        """
        Procesa una consulta:
          - Si use_retrieval es True y se ha proporcionado un document_retriever,
            se utiliza la búsqueda en documentos locales.
          - De lo contrario, se procesa la consulta de forma directa usando la nueva cadena LCEL.
        """
        if use_retrieval and self.document_retriever:
            return self.document_retriever.retrieve(query)
        # Invoca la nueva cadena LCEL pasando un diccionario con el valor de "query"
        answer = self.chain.invoke({"query": query})
        return QueryResponse(
            answer=answer,
            sources=[],
            confidence=0.85,
        )


def get_query_service():
    """
    Método auxiliar para inyectar QueryService en rutas de FastAPI.
    """
    from app.services.document_retriever import DocumentRetriever

    retriever = DocumentRetriever("documents")
    return QueryService(document_retriever=retriever)
