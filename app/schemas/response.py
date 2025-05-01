# app/schemas/response.py
from typing import List, Dict, Any
from pydantic import BaseModel


class Document(BaseModel):
    """
    Representación de un fragmento/documento recuperado.
    """

    id: str
    metadata: Dict[str, Any]
    page_content: str
    type: str = "document"


class QueryResponse(BaseModel):
    """
    DTO de respuesta para la API: texto generado y contexto de documentos.
    """

    answer: str
    context: List[Document]
