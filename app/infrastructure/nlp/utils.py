import spacy
from typing import List
from spacy_langdetect import LanguageDetector
from spacy.language import Language

# Cargar el modelo de spaCy en español
nlp = spacy.load("es_core_news_md")


# Añadir el detector de idioma (opcional, útil si el sistema detecta texto en otros idiomas)
@Language.factory("language_detector")
def get_lang_detector(nlp, name):
    return LanguageDetector()


nlp.add_pipe("language_detector", last=True)


def extract_keywords(text: str, min_len=3) -> List[str]:
    doc = nlp(text)
    keywords = set()

    for token in doc:
        if token.pos_ in {"NOUN", "PROPN", "ADJ"} and len(token.text) >= min_len:
            lemma = token.lemma_.lower()
            keywords.add(lemma)

    return list(keywords)


def extract_named_entities(text: str) -> List[str]:
    doc = nlp(text)
    return [ent.text for ent in doc.ents]


def split_into_sentences(text: str) -> List[str]:
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]
