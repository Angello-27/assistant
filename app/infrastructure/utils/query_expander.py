# app/infrastructure/utils/query_expander.py
import os
import re
import json
from typing import Dict, List
from app.domain.repositories.iquery_expander import IQueryExpander


class QueryExpander(IQueryExpander):
    """
    Expande una consulta utilizando un diccionario de jerga boliviana
    para sustituir sinónimos coloquiales por términos técnicos.

    Implementa el contrato definido en app/domain/repositories/iquery_expander.py
    """

    def __init__(self, file_path: str = None):
        # Determina la ruta al JSON de jerga
        if file_path:
            self.file_path = file_path
        else:
            # Asume que jerga_boliviana.json está en la raíz del proyecto
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            self.file_path = os.path.join(base_dir, "jerga_boliviana.json")
        # Carga el diccionario al inicializar la clase
        self._synonyms_map = self._load_jerga_dict()

    def _load_jerga_dict(self) -> Dict[str, List[str]]:
        """
        Lee y devuelve el mapa de jerga desde el archivo JSON.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Archivo de jerga no encontrado: {self.file_path}")
        with open(self.file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def expand(self, query: str) -> str:
        """
        Reemplaza en la consulta cada sinónimo boliviano por su término técnico
        correspondiente, según el diccionario cargado.
        """
        # Lowercase para coincidencias insensibles a mayúsculas
        expanded = query
        # Construye un mapping inverso: palabra coloquial -> técnica
        replacement_map = {
            syn.lower(): base
            for base, syns in self._synonyms_map.items()
            for syn in syns
        }

        # Para cada jerga detectada, reemplazar en el texto
        for colloquial, technical in replacement_map.items():
            # Patrón word boundary para coincidencias exactas
            pattern = rf"\b{re.escape(colloquial)}\b"
            expanded = re.sub(pattern, technical, expanded, flags=re.IGNORECASE)

        return expanded
