# app/entrypoints/http/dependencies.py
from fastapi import Depends
from app.domain.repositories.idocument_retriever import IDocumentRetriever
from app.domain.repositories.idocument_loader import IDocumentLoader
from app.domain.repositories.ivector_repository import IVectorRepository
from app.domain.repositories.irag_engine import IRetrievalEngine
from app.infrastructure.persistence.document_retriever import DocumentRetriever
from app.infrastructure.persistence.document_loader import FileSystemDocumentLoader
from app.infrastructure.persistence.faiss_repository import FAISSRepository
from app.infrastructure.langchain.retrieval_engine import RagEngine
from app.infrastructure.utils.query_expander import QueryExpander
from app.domain.repositories.iquery_expander import IQueryExpander
from app.usecases.query_interactor import QueryInteractor


def get_document_loader() -> IDocumentLoader:
    """
    Provee la implementación por defecto de IDocumentLoader.
    Carga archivos .txt desde el sistema de ficheros.
    """
    return FileSystemDocumentLoader()


def get_vector_repository() -> IVectorRepository:
    """
    Provee la implementación por defecto de IVectorRepository.
    Usa FAISS como backend de vectores.
    """
    return FAISSRepository()


def get_document_retriever(
    loader: IDocumentLoader = Depends(get_document_loader),
    vector_repo: IVectorRepository = Depends(get_vector_repository),
) -> IDocumentRetriever:
    """
    Provee DocumentRetriever inyectando:
      - loader     : IDocumentLoader (para leer textos)
      - vector_repo: IVectorRepository (para construir/cargar FAISS)
    """
    # Directorio donde residen los documentos legales
    documents_dir = "documents"
    return DocumentRetriever(
        documents_directory=documents_dir,
        loader=loader,
        vector_repo=vector_repo,
    )


def get_query_expander() -> IQueryExpander:
    """
    Provee la implementación por defecto de IQueryExpander.
    Usa un diccionario de jerga boliviana para normalizar la consulta.
    """
    return QueryExpander()


def get_rag_engine(
    vector_store: IVectorRepository = Depends(get_vector_repository),
) -> IRetrievalEngine:
    """
    Instancia y cachea el RagEngine:
      - Recibe un IVectorRepository para obtener el FAISS index.
      - Construye internamente la cadena RAG (prompt, LLM, retriever).
    """
    return RagEngine(vector_store)


def get_query_interactor(
    retriever: IDocumentRetriever = Depends(get_document_retriever),
    expander: IQueryExpander = Depends(get_query_expander),
    rag_engine: IRetrievalEngine = Depends(get_rag_engine),
) -> QueryInteractor:
    """
    Construye el caso de uso QueryInteractor con sus dependencias:
      1. DocumentRetriever (busca y fragmenta documentos)
      2. QueryExpander    (normaliza jerga de la consulta)
      3. RagEngine        (realiza el pipeline RAG y genera la respuesta)
    """
    return QueryInteractor(
        document_retriever=retriever,
        query_expander=expander,
        retrieval_engine=rag_engine,
    )
