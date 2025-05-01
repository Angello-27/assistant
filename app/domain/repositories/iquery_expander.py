# app/domain/repositories/iquery_expander.py
from abc import ABC, abstractmethod


class IQueryExpander(ABC):
    """
    Contrato para servicios que expanden o normalizan la consulta del usuario.
    """

    @abstractmethod
    def expand(self, query: str) -> str:
        """
        Recibe la consulta original y devuelve una versión expandida
        o normalizada para mejorar la recuperación de información.
        """
        pass
