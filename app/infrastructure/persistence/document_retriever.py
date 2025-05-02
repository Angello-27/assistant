# app/infrastructure/persistence/document_retriever.py
from typing import List
from app.domain.repositories.idocument_retriever import IDocumentRetriever
from app.domain.repositories.idocument_loader import IDocumentLoader
from app.infrastructure.persistence.faiss_repository import FAISSRepository
from app.infrastructure.persistence.document_loader import FileSystemDocumentLoader
from app.infrastructure.utils.nlp.splitter import split_document_spacy
from app.domain.entities.fragment import Fragment
from app.infrastructure.langchain.retrieval_engine import process_query_with_retrieval


class DocumentRetriever(IDocumentRetriever):
    """
    Implementación de IDocumentRetriever usando FAISS + RAG.
    Inyecta un loader de documentos y un repositorio FAISS.
    """

    def __init__(
        self,
        documents_directory: str,
        loader: IDocumentLoader | None = None,
        faiss_repo: FAISSRepository | None = None,
    ):
        self.documents_directory = documents_directory
        # si no se inyecta, usamos la implementación por defecto
        self.loader = loader or FileSystemDocumentLoader()
        self.repo = faiss_repo or FAISSRepository()
        # intentar cargar índice en disco
        try:
            self.vector_store = self.repo.load_vectorstore()
        except FileNotFoundError:
            self.vector_store = self._prepare_vectorstore()

    def _prepare_vectorstore(self):
        """
        1. Carga documentos crudos via IDocumentLoader
        2. Fragmenta cada texto con spaCy/Regex
        3. Construye y persiste el vectorstore FAISS
        """
        raw_docs = self.loader.load_all(self.documents_directory)
        fragments: List[Fragment] = []
        for raw in raw_docs:
            content = raw.page_content
            source = raw.metadata.get("source", "documento")
            fragments.extend(split_document_spacy(content, source))
        return self.repo.build_vectorstore(fragments)

    def retrieve(self, query: str):
        """
        Llama al motor RAG para procesar la consulta contra el vector store.
        """
        return process_query_with_retrieval(query, self.vector_store)
