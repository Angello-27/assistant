# app/domain/repositories/iquery_expander.py
from abc import ABC, abstractmethod


class IQueryExpander(ABC):
    """
    Contrato para servicios que expanden o normalizan la consulta del usuario.
    (p. ej. jerga boliviana).
    """

    @abstractmethod
    def expand(self, query: str) -> str:
        """
        Dada una consulta simple, retorna una versión enriquecida
        reemplazando jerga o sinónimos por términos técnicos.
        """
        ...
