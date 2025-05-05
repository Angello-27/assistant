# app/infrastructure/utils/nlp/splitter.py
import re
import uuid
from typing import List
from app.domain.entities.fragment import Fragment
from app.infrastructure.utils.nlp.utils import extract_keywords
from langchain_text_splitters import RecursiveCharacterTextSplitter


def is_structured(text: str) -> bool:
    """
    Detecta si el bloque contiene al menos un 'Artículo N°'.
    """
    return bool(re.search(r"Artículo\s+\d+", text))


def generate_id(doc_name: str, index: int) -> str:
    """
    Genera un ID único combinando nombre de documento, índice y UUID corto.
    """
    return f"{doc_name.replace('.txt','')}-{index}-{uuid.uuid4().hex[:8]}"


def normalize_chunk(chunk: str) -> str:
    """
    Normaliza espacios y saltos de línea en un solo párrafo.
    """
    return chunk.strip().replace("\n", " ").replace("  ", " ").strip()


def extract_articles(text: str) -> List[str]:
    """
    Extrae cada Artículo completo con su número y contenido.
    """
    pattern = re.compile(r"(Artículo\s+\d+.*?)(?=(\nArtículo\s+\d+|\Z))", re.DOTALL)
    return [m.group(1) for m in pattern.finditer(text)]


def split_document_spacy(text: str, doc_name: str) -> List[Fragment]:
    """
    Divide texto en fragmentos:
      1. LangChain para chunks grandes.
      2. Regex para separar Artículos.
      3. Limpieza y extracción de keywords con spaCy.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=300)
    chunks = splitter.split_text(text)
    index = 0
    fragments: List[Fragment] = []

    for chunk in chunks:
        block = chunk.strip()
        if is_structured(block):
            for art in extract_articles(block):
                clean = normalize_chunk(art)
                if not clean:
                    continue
                fragment_id = generate_id(doc_name, index)
                tags = extract_keywords(clean)
                fragments.append(
                    Fragment(
                        id=fragment_id,
                        content=clean,
                        source=doc_name,
                        position=index,
                        tags=tags,
                        synonyms=[],
                    )
                )
                index += 1
        else:
            # Bloque libre (e.g. decretos, intro)
            clean = normalize_chunk(block)
            if clean:
                fragment_id = generate_id(doc_name, index)
                tags = extract_keywords(clean)
                fragments.append(
                    Fragment(
                        id=fragment_id,
                        content=clean,
                        source=doc_name,
                        position=index,
                        tags=tags,
                        synonyms=[],
                    )
                )
                index += 1

    return fragments
