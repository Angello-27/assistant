# app/domain/repositories/irag_engine.py
from abc import ABC, abstractmethod
from app.schemas.response import QueryResponse


class IRetrievalEngine(ABC):
    """
    Contrato para cualquier implementación de ‘motor RAG’.
    """

    @abstractmethod
    def retrieve(self, query: str) -> QueryResponse:
        """
        Dada una consulta, ejecuta el pipeline RAG completo
        (recuperación + LLM + memoria conversacional) y devuelve un QueryResponse.
        """
        pass
