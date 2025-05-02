# app/domain/repositories/idocument_retriever.py
from abc import ABC, abstractmethod
from langchain_community.vectorstores import FAISS


class IDocumentRetriever(ABC):
    """
    Contrato para proveedores de vector store de documentos.
    Sólo expone get_vector_store(), que devuelve la instancia FAISS ya cargada
    o construida.
    """

    @abstractmethod
    def get_vector_store(self) -> FAISS:
        """
        Devuelve el FAISS vector store, construyéndolo o cargándolo de persistencia.
        """
        ...
