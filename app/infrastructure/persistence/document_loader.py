# app/infrastructure/persistence/document_loader.py

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from typing import List
from langchain.document_loaders import Document
from app.domain.repositories.idocument_loader import IDocumentLoader


class DocumentLoader(IDocumentLoader):
    """
    Implementación de IDocumentLoader que utiliza LangChain Community
    para cargar .txt desde un directorio.
    """

    def load_all(self, directory_path: str) -> List[Document]:
        loader = DirectoryLoader(
            directory_path,
            glob="*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        return loader.load()
