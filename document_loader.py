from langchain_community.document_loaders import DirectoryLoader, TextLoader
from typing import List


def load_documents_from_directory(directory_path: str) -> List:
    """
    Carga todos los documentos de texto del directorio especificado.
    Se asume que los documentos están en formato .txt.
    """
    loader = DirectoryLoader(
        directory_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    return loader.load()
