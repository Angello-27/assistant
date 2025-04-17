import re
import uuid
from typing import List


def generate_id(doc_name: str, index: int) -> str:
    return f"{doc_name.replace('.txt', '')}-{index}-{uuid.uuid4().hex[:8]}"


# ⬇ Nuevos métodos de segmentación legal (por bloques jerárquicos)
def split_by_titles(text: str) -> List[str]:
    return re.split(
        r"(Título\s+[IVXLCDM]+\b.*?)(?=\n(Título\s+[IVXLCDM]+\b)|\Z)",
        text,
        flags=re.DOTALL,
    )


def split_by_chapters(text: str) -> List[str]:
    return re.split(
        r"(Capítulo\s+[IVXLCDM]+\b.*?)(?=\n(Capítulo\s+[IVXLCDM]+\b)|\Z)",
        text,
        flags=re.DOTALL,
    )


def split_by_articles(text: str) -> List[str]:
    return re.findall(
        r"(Artículo\s+\d+\.?.*?)(?=(\nArtículo\s+\d+\.?)|\Z)", text, flags=re.DOTALL
    )
