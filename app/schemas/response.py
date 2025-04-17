from typing import List, Any, Dict
from pydantic import BaseModel


class Document(BaseModel):
    id: str
    metadata: Dict[str, Any]
    page_content: str
    type: str = "document"


class QueryResponse(BaseModel):
    answer: str
    context: List[Document]
