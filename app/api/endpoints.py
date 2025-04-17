from fastapi import APIRouter, Depends, HTTPException
from app.services.query_service import get_query_service
from app.schemas.query import QueryRequest

router = APIRouter()


@router.post("/ask")
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
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
