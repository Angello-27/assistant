from fastapi import APIRouter, HTTPException
from app.models.query import QueryRequest
from app.services.query_service import QueryService
from app.services.document_retriever import DocumentRetriever

router = APIRouter()

# Instanciar DocumentRetriever con la ruta de tu carpeta de documentos
document_retriever = DocumentRetriever(documents_directory="documents")
# Instanciar QueryService inyectando el document_retriever
query_service = QueryService(document_retriever=document_retriever)


@router.post("/ask")
async def ask_question(request: QueryRequest, use_retrieval: bool = True):
    try:
        answer = query_service.query(request.query, use_retrieval=use_retrieval)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
