# app/domain/repositories/ivector_repository.py
from abc import ABC, abstractmethod
from langchain_community.vectorstores import FAISS
from app.domain.entities.fragment import Fragment
from typing import List


class IVectorRepository(ABC):
    """
    Contrato para cualquier repositorio de vectores (FAISS, Pinecone, etc.).
    """

    @abstractmethod
    def build(self, fragments: List[Fragment]) -> FAISS:
        """
        Construye y persiste el índice de vectores a partir de los fragmentos.
        """
        pass

    @abstractmethod
    def load(self) -> FAISS:
        """
        Carga el índice de vectores previamente persistido.
        """
        pass
