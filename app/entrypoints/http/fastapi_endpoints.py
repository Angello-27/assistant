# app/entrypoints/http/fastapi_endpoints.py
from fastapi import APIRouter, HTTPException, Depends
from app.schemas.query import QueryRequest
from app.schemas.response import QueryResponse
from app.usecases.query_interactor import QueryInteractor
from app.entrypoints.http.dependencies import get_query_interactor

router = APIRouter()


@router.post("/ask", response_model=QueryResponse)
async def ask_question(
    request: QueryRequest,
    interactor: QueryInteractor = Depends(get_query_interactor),
):
    """
    Endpoint principal: recibe la consulta, la expande y recupera la respuesta.
    Llama al caso de uso QueryInteractor.
    """
    try:
        return interactor.execute(request.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
