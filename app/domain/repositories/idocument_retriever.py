# app/domain/repositories/idocument_retriever.py
from abc import ABC, abstractmethod
from app.schemas.response import QueryResponse


class IDocumentRetriever(ABC):
    """
    Contrato para recuperadores de documentos.
    Define la operación de búsqueda/retrieval que devuelve un QueryResponse.
    """

    @abstractmethod
    def retrieve(self, query: str) -> QueryResponse:
        """
        Ejecuta la búsqueda de la consulta expandida y devuelve
        un objeto QueryResponse con la respuesta y el contexto.
        """
        ...
