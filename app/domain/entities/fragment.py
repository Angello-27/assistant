# app/domain/entities/fragment.py
from typing import List, Optional
from pydantic import BaseModel


class Fragment(BaseModel):
    """
    Entidad de dominio que representa un fragmento de texto (artículo o bloque libre).
    """

    id: str  # identificador único del fragmento
    content: str  # texto donde se basa la respuesta
    source: str  # nombre del archivo de origen
    section: Optional[str] = None  # sección/legal, si aplica
    position: Optional[int] = None  # orden para temas de paginación o historial
    tags: List[str] = []  # palabras clave extraídas automáticamente
    synonyms: List[str] = []  # sinónimos agregados manual/automáticamente
