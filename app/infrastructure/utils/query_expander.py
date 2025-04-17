import os
import re
import json
from typing import Dict

# Definir la ruta correcta al archivo JSON
FILE_PATH = os.path.join("app", "utils", "jerga_boliviana.json")


def cargar_diccionario_json(file_path: str) -> Dict[str, list[str]]:
    """Carga el diccionario de sinónimos desde un archivo JSON."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


# Diccionario especializado con jerga boliviana
JERGA_BOLIVIANA = cargar_diccionario_json(FILE_PATH)


def expand_query(query: str) -> str:
    """
    Expande una consulta agregando sinónimos bolivianos para mejorar la recuperación semántica
    y reformulando con expresiones regulares para adaptar el lenguaje.
    """
    query_lower = query.lower()
    expanded_query = query  # Se parte de la consulta original

    # Crear diccionario de reemplazo: sinónimo -> palabra técnica
    replacement = {
        syn: base_word
        for base_word, synonyms in JERGA_BOLIVIANA.items()
        for syn in synonyms
    }

    # Reemplazar cada palabra de jerga si aparece en la consulta
    for old_word, new_word in replacement.items():
        # Crear un patrón que capture todas las variaciones de la palabra clave (sinónimos)
        pattern = r"\b" + re.escape(old_word) + r"\b"
        # Reemplazamos la frase utilizando las reglas definidas en el diccionario de sinónimos
        expanded_query = re.sub(pattern, new_word, expanded_query, flags=re.IGNORECASE)

    # Ahora, con los sinónimos reemplazados, se puede agregar más contexto o estructura si es necesario
    return expanded_query
