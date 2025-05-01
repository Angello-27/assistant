# app/domain/repositories/idocument_loader.py
from abc import ABC, abstractmethod
from typing import List
from langchain.document_loaders import Document


class IDocumentLoader(ABC):
    """
    Port de dominio para cargar documentos desde una fuente (p.ej. ficheros).
    Permite desacoplar la lógica de negocio de la fuente de datos concreta.
    """

    @abstractmethod
    def load_all(self, directory_path: str) -> List[Document]:
        """
        Debe devolver todos los documentos crudos (por ejemplo, langchain.Document)
        contenidos en el directorio.
        """
        pass
