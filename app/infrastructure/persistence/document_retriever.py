# app/infrastructure/persistence/document_retriever.py
from typing import List
from app.domain.repositories.idocument_retriever import IDocumentRetriever
from app.domain.repositories.idocument_loader import IDocumentLoader
from app.domain.repositories.ivector_repository import IVectorRepository
from app.infrastructure.persistence.document_loader import FileSystemDocumentLoader
from app.infrastructure.persistence.faiss_repository import FAISSRepository
from app.infrastructure.utils.nlp.splitter import split_document_spacy
from app.domain.entities.fragment import Fragment


class DocumentRetriever(IDocumentRetriever):
    """
    Implementación de IDocumentRetriever usando un IVectorRepository (FAISS).
    Se inyecta además un loader de documentos (IDocumentLoader).
    """

    def __init__(
        self,
        documents_directory: str,
        loader: IDocumentLoader | None = None,
        vector_repo: IVectorRepository | None = None,
    ):
        self.documents_directory = documents_directory
        # si no se inyecta, usamos la implementación por defecto
        self.loader: IDocumentLoader = loader or FileSystemDocumentLoader()
        self.repo: IVectorRepository = vector_repo or FAISSRepository()

        # Intentar cargar el índice de vectores desde disco; si no existe, reconstruirlo
        try:
            self.vector_store = self.repo.load()
            print("[Retriever] Vector store cargado desde disco.")
        except FileNotFoundError:
            print("[Retriever] Vector store no encontrado, reconstruyendo...")
            self.vector_store = self._prepare_vectorstore()

    def _prepare_vectorstore(self):
        """
        1. Carga documentos crudos vía IDocumentLoader.load_all()
        2. Fragmenta cada texto con spaCy/Regex
        3. Llama a IVectorRepository.build() para crear y persistir el índice
        """
        raw_docs = self.loader.load_all(self.documents_directory)
        fragments: List[Fragment] = []
        for raw in raw_docs:
            content = raw.page_content
            source = raw.metadata.get("source", "documento")
            fragments.extend(split_document_spacy(content, source))
        return self.repo.build(fragments)

    def get_vector_store(self):
        """
        Devuelve el vector store.
        """
        return self.vector_store
