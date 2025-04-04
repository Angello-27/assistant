from fastapi import APIRouter, HTTPException
from app.models.query import QueryRequest
from app.services.openai_service import process_query

router = APIRouter()


@router.post("/ask")
async def ask_question(request: QueryRequest):
    try:
        answer = process_query(request.query)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
