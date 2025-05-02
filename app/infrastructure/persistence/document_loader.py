# app/infrastructure/persistence/document_loader.py
from typing import List
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.docstore.document import Document as LcDocument
from app.domain.repositories.idocument_loader import IDocumentLoader


class FileSystemDocumentLoader(IDocumentLoader):
    """
    Implementación de IDocumentLoader que carga todos los .txt de un directorio
    usando LangChain DirectoryLoader + TextLoader.
    """

    def load_all(self, directory_path: str) -> List[LcDocument]:
        loader = DirectoryLoader(
            directory_path,
            glob="*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        return loader.load()
