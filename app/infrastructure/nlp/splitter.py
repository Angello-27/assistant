import re
import uuid
from typing import List
from app.domain.entities.fragment import Fragment
from app.infrastructure.nlp.utils import extract_keywords
from langchain_text_splitters import RecursiveCharacterTextSplitter


def is_structured(text: str) -> bool:
    return "Artículo" in text and re.search(r"Artículo\s+\d+", text)


def generate_id(doc_name: str, index: int) -> str:
    return f"{doc_name.replace('.txt', '')}-{index}-{uuid.uuid4().hex[:8]}"


def normalize_chunk(chunk: str) -> str:
    return chunk.strip().replace("\n", " ").replace("  ", " ").strip()


# -- Extractores puros --
def extract_articles(text: str) -> List[str]:
    pattern = re.compile(r"(Artículo\s+\d+.*?)(?=(\nArtículo\s+\d+|\Z))", re.DOTALL)
    return [m.group(1) for m in pattern.finditer(text)]


# Proceso combinado: LangChain → spaCy → enriquecimiento
def split_document_spacy(text: str, doc_name: str) -> List[Fragment]:
    """
    Aplica una división jerárquica con LangChain + NLP.
    split_strategy: función que define cómo se divide internamente (por artículos, capítulos, etc.)
    """
    # Configura el text splitter: chunk_size es el tamaño máximo del fragmento,
    # chunk_overlap es la cantidad de caracteres que se solapan entre fragmentos.
    splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=300)
    base_chunks = splitter.split_text(text)

    index = 0
    fragments: List[Fragment] = []

    for base in base_chunks:
        base = base.strip()

        # Si tiene estructura con artículos, dividirlos
        if is_structured(base):
            # dentro de cada bloque, dividir por Artículos o Capítulos
            for raw_article in extract_articles(base):
                if not raw_article:
                    print("⚠️ Artículo vacío o None detectado. Saltando.")
                    continue

                # Podrías alternar split_by_chapters, split_by_titles, etc.
                clean_text = normalize_chunk(
                    raw_article if isinstance(raw_article, str) else raw_article[0]
                )
                print("📄 Artículo extraído:", clean_text[:120])
                if not clean_text:
                    print("⚠️ Artículo normalizado está vacío. Saltando.")
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
        else:
            # ✅ Si no tiene estructura legal, guardarlo como bloque libre
            print(
                "📄 Documento no estructurado detectado. Procesando como bloque libre."
            )
            clean_text = normalize_chunk(base)
            if clean_text:
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
