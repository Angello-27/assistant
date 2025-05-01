# app/domain/repositories/idocument_retriever.py
from abc import ABC, abstractmethod
from app.schemas.response import QueryResponse


class IDocumentRetriever(ABC):
    """
    Contrato para cualquier servicio de recuperación de documentos.
    """

    @abstractmethod
    def retrieve(self, query: str) -> QueryResponse:
        """
        Retorna un QueryResponse con la respuesta y contexto de documentos.
        """
        pass
