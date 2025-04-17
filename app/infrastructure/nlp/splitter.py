import re
import uuid
from typing import List
from app.domain.entities.fragment import Fragment
from app.infrastructure.nlp.nlp_utils import extract_keywords
from langchain_text_splitters import RecursiveCharacterTextSplitter


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


def normalize_chunk(chunk: str) -> str:
    return chunk.strip().replace("\n", " ").replace("  ", " ").strip()


# 🔁 Proceso combinado: LangChain → spaCy → enriquecimiento
def split_document_spacy(text: str, doc_name: str) -> List[Fragment]:
    # 1. Primera capa: dividir con LangChain si el texto es muy largo
    splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=300)
    base_chunks = splitter.split_text(text)

    fragments: List[Fragment] = []
    index = 0

    for base in base_chunks:
        # 2. Segunda capa: dentro de cada bloque, dividir por Artículos o Capítulos
        for article in split_by_articles(
            base
        ):  # Podrías alternar split_by_chapters, split_by_titles, etc.
            clean_text = normalize_chunk(
                article if isinstance(article, str) else article[0]
            )
            if not clean_text:
                continue

            fragment_id = generate_id(doc_name, index)
            keywords = extract_keywords(clean_text)

            fragments.append(
                Fragment(
                    id=fragment_id,
                    content=clean_text,
                    source=doc_name,
                    position=index,
                    tags=keywords,
                    synonyms=[],
                )
            )
            index += 1

    return fragments
