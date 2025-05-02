# app/domain/repositories/idocument_loader.py
from abc import ABC, abstractmethod
from typing import List
from langchain.docstore.document import Document as LcDocument


class IDocumentLoader(ABC):
    """
    Port de dominio para cargar documentos desde una fuente (p.ej. ficheros).
    Permite desacoplar la lógica de negocio de la fuente de datos concreta.
    """

    @abstractmethod
    def load_all(self, directory_path: str) -> List[LcDocument]:
        """
        Debe devolver todos los documentos crudos (langchain.docstore.document.Document)
        contenidos en el directorio.
        """
        ...
