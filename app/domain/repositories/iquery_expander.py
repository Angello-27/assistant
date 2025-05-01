# app/domain/repositories/iquery_expander.py
from abc import ABC, abstractmethod


class IQueryExpander(ABC):
    """
    Contrato para servicios que expanden o normalizan la consulta del usuario.
    """

    @abstractmethod
    def expand(self, query: str) -> str:
        """
        Recibe una consulta y devuelve una versión expandida
        (sinónimos, jerga, limpieza).
        """
        pass
