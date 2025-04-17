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
    expanded_parts = [query]

    for word, synonyms in JERGA_BOLIVIANA.items():
        # Crear un patrón que capture todas las variaciones de la palabra clave (sinónimos)
        pattern = (
            r"\b(?:" + "|".join([re.escape(s) for s in [word] + synonyms]) + r")\b"
        )

        # Si alguna de las palabras o sinónimos aparece en la consulta, expandimos la consulta
        if re.search(pattern, query_lower):
            # Reemplazar las coincidencias de los sinónimos por un formato más técnico
            replacement = {
                "paco": "policía de tránsito",
                "engrapo": "detener el vehículo con grapa en el neumático",
                "boleta": "infracción de tránsito",
                "centro": "zona céntrica de la ciudad",
                "banco": "entidad financiera",
                "micro": "microbús",
                "transito": "regulaciones de tránsito",
                # Aquí podrías seguir añadiendo más sinónimos con sus sustituciones
            }

            # Reemplazamos la frase utilizando las reglas definidas en el diccionario de sinónimos
            for old_word, new_word in replacement.items():
                expanded_query = re.sub(
                    r"\b" + re.escape(old_word) + r"\b", new_word, expanded_query
                )

    # Ahora, con los sinónimos reemplazados, se puede agregar más contexto o estructura si es necesario
    return expanded_query
