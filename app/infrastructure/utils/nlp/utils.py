# File: app/infrastructure/utils/nlp/utils.py
import spacy
from spacy_langdetect import LanguageDetector
from spacy.language import Language
from typing import List

# Carga modelo español para análisis de texto
nlp = spacy.load("es_core_news_md")


@Language.factory("language_detector")
def get_lang_detector(nlp, name):
    """
    Fabrica y añade el detector de idioma al pipeline.
    """
    return LanguageDetector()


# Agrega el detector al final del pipeline
nlp.add_pipe("language_detector", last=True)


def extract_keywords(text: str, min_len: int = 3) -> List[str]:
    """
    Extrae lemas de sustantivos, nombres propios y adjetivos
    que superen la longitud mínima, solo si el texto está en español.
    """
    doc = nlp(text)
    # Verificar idioma
    lang_meta = doc._.language  # {'language': 'es', 'score': 0.99}
    if lang_meta.get("language") != "es" or lang_meta.get("score", 0) < 0.8:
        return []
    keywords = set()
    for token in doc:
        if token.pos_ in {"NOUN", "PROPN", "ADJ"} and len(token.text) >= min_len:
            keywords.add(token.lemma_.lower())
    return list(keywords)
