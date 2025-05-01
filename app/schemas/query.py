# app/schemas/query.py
from pydantic import BaseModel


class QueryRequest(BaseModel):
    """
    DTO para la petición /ask: solo lleva el texto de la consulta.
    """

    query: str
