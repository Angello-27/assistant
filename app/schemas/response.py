from pydantic import BaseModel


class QueryResponse(BaseModel):
    answer: str
    sources: list[str] = []
    confidence: float = 0.85
