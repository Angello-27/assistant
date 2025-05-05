# File: app/infrastructure/utils/nlp/splitter.py
import re
import uuid
from typing import List
from app.domain.entities.fragment import Fragment
from app.infrastructure.utils.nlp.utils import extract_keywords, nlp


def split_into_paragraphs(text: str) -> List[str]:
    """
    Divide el texto completo en párrafos separados por líneas en blanco.
    """
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def generate_id(doc_name: str, index: int) -> str:
    """
    Genera un ID único combinando nombre de documento, índice y UUID corto.
    """
    return f"{doc_name.replace('.txt','')}-{index}-{uuid.uuid4().hex[:8]}"


def process_article_block(paragraph: str, doc_name: str, index: int) -> List[Fragment]:
    """
    Extrae fragmentos de un párrafo que contenga artículos numerados y variantes,
    manteniendo sus incisos dentro del mismo bloque.
    Soporta encabezados como:
      - Artículo 1°
      - Artículo Único
      - Artículo Transitorio
      - Artículo Final
      - Disposición Final
    """
    fragments: List[Fragment] = []
    # Patrón para encabezados de artículo o disposición final
    article_header = r"(?:Art[ií]culo(?:\s+(?:Único|Transitorio|Final))?(?:\s+\d+°?)?|Disposición\s+Final)"
    # Capturar desde el encabezado hasta justo antes del siguiente encabezado o fin de texto
    pattern = re.compile(
        rf"({article_header}[\s\S]*?)(?=(?:\n{article_header}|$))",
        re.IGNORECASE,
    )

    for match in pattern.finditer(paragraph):
        block = match.group(1).strip()
        # Filtrar idiomas no-español
        doc = nlp(block)
        lang_meta = doc._.language
        if lang_meta.get("language") != "es" or lang_meta.get("score", 0) < 0.8:
            continue

        # Extraer etiquetas y sinónimos
        tags = extract_keywords(block)
        synonyms = []

        frag_id = generate_id(doc_name, index)
        fragments.append(
            Fragment(
                id=frag_id,
                content=block,
                source=doc_name,
                position=index,
                tags=tags,
                synonyms=synonyms,
            )
        )
        index += 1
    return fragments


def process_free_block(paragraph: str, doc_name: str, index: int) -> Fragment:
    """
    Procesa un párrafo libre sin encabezados especiales.
    """
    # Filtrar idioma
    doc = nlp(paragraph)
    lang_meta = doc._.language
    if lang_meta.get("language") != "es" or lang_meta.get("score", 0) < 0.8:
        return None

    clean = paragraph.replace("\n", " ").strip()
    tags = extract_keywords(clean)
    synonyms = []

    frag_id = generate_id(doc_name, index)
    return Fragment(
        id=frag_id,
        content=clean,
        source=doc_name,
        position=index,
        tags=tags,
        synonyms=synonyms,
    )


def split_document_spacy(text: str, doc_name: str) -> List[Fragment]:
    """
    Divide el texto en fragmentos:
      1) Separa en párrafos.
      2) Si un párrafo inicia con 'Artículo' o 'Disposición Final', trata todos
         los artículos/variantes como bloques únicos con incisos.
      3) En otro caso, crea fragmento libre.
    """
    paragraphs = split_into_paragraphs(text)
    fragments: List[Fragment] = []
    index = 0

    for para in paragraphs:
        # Detecta encabezado de artículo o disposición final
        if re.match(r"^(?:Art[ií]culo|Disposición\s+Final)", para, re.IGNORECASE):
            arts = process_article_block(para, doc_name, index)
            fragments.extend(arts)
            index += len(arts)
        else:
            frag = process_free_block(para, doc_name, index)
            if frag:
                fragments.append(frag)
                index += 1

    return fragments
