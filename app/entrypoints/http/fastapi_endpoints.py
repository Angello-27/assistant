# app/entrypoints/http/fastapi_endpoints.py
from fastapi import APIRouter, Depends, HTTPException
from app.schemas.query import QueryRequest
from app.schemas.response import QueryResponse
from app.usecases.query_interactor import QueryInteractor
from app.domain.repositories.idocument_retriever import IDocumentRetriever
from app.domain.repositories.iquery_expander import IQueryExpander
from app.infrastructure.persistence.document_retriever import DocumentRetriever
from app.infrastructure.utils.query_expander import QueryExpander

router = APIRouter()


def get_query_interactor(
    retriever: IDocumentRetriever = Depends(lambda: DocumentRetriever("documents")),
    expander: IQueryExpander = Depends(QueryExpander),
) -> QueryInteractor:
    """
    Dependencia: construye el interactor con su retriever y expander.
    """
    return QueryInteractor(document_retriever=retriever, query_expander=expander)


@router.post("/ask", response_model=QueryResponse)
async def ask_question(
    request: QueryRequest,
    interactor: QueryInteractor = Depends(get_query_interactor),
):
    """
    Endpoint principal que procesa una consulta legal.
    Llama al caso de uso QueryInteractor.
    """
    try:
        # Ejecuta el caso de uso, devuelve QueryResponse
        return interactor.execute(request.query)
    except Exception as e:
        # Captura errores y responde 500 con detalle
        raise HTTPException(status_code=500, detail=str(e))
