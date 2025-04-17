from typing import List, Optional
from pydantic import BaseModel


class Fragment(BaseModel):
    id: str  # identificador único
    content: str  # el texto del fragmento
    source: str  # nombre del documento original
    section: Optional[str] = None  # opcional: sección del documento (si aplica)
    position: Optional[int] = None  # orden en el documento
    tags: List[str] = []  # palabras clave (extraídas con NLP)
    synonyms: List[str] = []  # sinónimos asociados
