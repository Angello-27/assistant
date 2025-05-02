# app/entrypoints/http/dependencies.py
from fastapi import Depends
from app.domain.repositories.idocument_retriever import IDocumentRetriever
from app.domain.repositories.idocument_loader import IDocumentLoader
from app.domain.repositories.ivector_repository import IVectorRepository
from app.infrastructure.persistence.document_retriever import DocumentRetriever
from app.infrastructure.persistence.document_loader import FileSystemDocumentLoader
from app.infrastructure.persistence.faiss_repository import FAISSRepository
from app.infrastructure.utils.query_expander import QueryExpander
from app.domain.repositories.iquery_expander import IQueryExpander
from app.usecases.query_interactor import QueryInteractor


def get_document_loader() -> IDocumentLoader:
    """
    Provee la implementación por defecto de IDocumentLoader
    """
    return FileSystemDocumentLoader()


def get_vector_repository() -> IVectorRepository:
    """
    Provee la implementación por defecto de IVectorRepository
    """
    return FAISSRepository()


def get_document_retriever(
    loader: IDocumentLoader = Depends(get_document_loader),
    vector_repo: IVectorRepository = Depends(get_vector_repository),
) -> IDocumentRetriever:
    """
    Provee DocumentRetriever inyectando:
     - loader   : IDocumentLoader
     - vector_repo: IVectorRepository
    """
    # El directorio 'documents' es fijo; si prefieres puedes parametrizarlo
    return DocumentRetriever(
        documents_directory="documents",
        loader=loader,
        vector_repo=vector_repo,
    )


def get_query_expander() -> IQueryExpander:
    """
    Provee la implementación por defecto de IQueryExpander (jerga boliviana).
    """
    return QueryExpander()


def get_query_interactor(
    retriever: IDocumentRetriever = Depends(get_document_retriever),
    expander: IQueryExpander = Depends(get_query_expander),
) -> QueryInteractor:
    """
    Construye el caso de uso QueryInteractor con sus dependencias:
     - DocumentRetriever
     - QueryExpander
    """
    return QueryInteractor(
        document_retriever=retriever,
        query_expander=expander,
    )
