# app/infrastructure/utils/nlp/utils.py
import spacy
from typing import List
from spacy_langdetect import LanguageDetector
from spacy.language import Language

# Carga modelo español (asegúrate de haberlo instalado)
nlp = spacy.load("es_core_news_md")


@Language.factory("language_detector")
def get_lang_detector(nlp, name):
    """
    Fabrica y añade el detector de idioma.
    """
    return LanguageDetector()


nlp.add_pipe("language_detector", last=True)


def extract_keywords(text: str, min_len: int = 3) -> List[str]:
    """
    Extrae lemas de sustantivos, nombres propios y adjetivos
    que superen la longitud mínima.
    """
    doc = nlp(text)
    keywords = set()
    for token in doc:
        if token.pos_ in {"NOUN", "PROPN", "ADJ"} and len(token.text) >= min_len:
            keywords.add(token.lemma_.lower())
    return list(keywords)


def extract_named_entities(text: str) -> List[str]:
    """
    Retorna entidades nombradas detectadas en el texto.
    """
    doc = nlp(text)
    return [ent.text for ent in doc.ents]


def split_into_sentences(text: str) -> List[str]:
    """
    Divide un párrafo en oraciones completas.
    """
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]
