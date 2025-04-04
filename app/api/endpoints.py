from fastapi import APIRouter, HTTPException
from app.models.query import QueryRequest
from app.services.transit_knowledge_service import TransitKnowledgeService

router = APIRouter()

# Instancia el servicio, indicando el directorio de documentos (asegúrate de que la carpeta 'documents' esté en la raíz del proyecto)
knowledge_service = TransitKnowledgeService(documents_directory="documents")


@router.post("/ask")
async def ask_question(request: QueryRequest, use_retrieval: bool = True):
    try:
        answer = knowledge_service.query(request.query, use_retrieval=use_retrieval)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
