from fastapi import APIRouter, Depends, HTTPException
from app.usecases.query_interactor import get_query_service
from app.schemas.query import QueryRequest
from app.schemas.response import QueryResponse

router = APIRouter()


@router.post("/ask", response_model=QueryResponse)
async def ask_question(
    request: QueryRequest,
    use_retrieval: bool = True,
    query_service=Depends(get_query_service),
):
    """
    Endpoint principal que procesa una consulta legal.
    Utiliza el servicio QueryService para generar una respuesta basada en documentos relevantes.
    """
    try:
        answer = query_service.query(request.query, use_retrieval=use_retrieval)
        return answer  #  Ya es un QueryResponse
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
