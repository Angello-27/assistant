from typing import List, Callable
from app.domain.entities.fragment import Fragment
from app.infrastructure.nlp.utils import extract_keywords
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.infrastructure.utils.generator import generate_id, split_by_articles


def normalize_chunk(chunk: str) -> str:
    return chunk.strip().replace("\n", " ").replace("  ", " ").strip()


# Proceso combinado: LangChain → spaCy → enriquecimiento
def split_document_spacy(
    text: str,
    doc_name: str,
    split_strategy: Callable[[str], List[str]] = split_by_articles,
) -> List[Fragment]:
    """
    Aplica una división jerárquica con LangChain + NLP.
    split_strategy: función que define cómo se divide internamente (por artículos, capítulos, etc.)
    """
    # Configura el text splitter: chunk_size es el tamaño máximo del fragmento,
    # chunk_overlap es la cantidad de caracteres que se solapan entre fragmentos.
    splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=300)
    base_chunks = splitter.split_text(text)

    fragments: List[Fragment] = []
    index = 0

    for base in base_chunks:
        # Segunda capa: dentro de cada bloque, dividir por Artículos o Capítulos
        for article in split_strategy(base):
            if not article:
                print("⚠️ Artículo vacío o None detectado. Saltando.")
                continue

            # Podrías alternar split_by_chapters, split_by_titles, etc.
            clean_text = normalize_chunk(
                article if isinstance(article, str) else article[0]
            )
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

    return fragments
