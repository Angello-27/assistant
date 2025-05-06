# app/infrastructure/utils/topic_detector.py
from typing import List, Set
from app.infrastructure.utils.nlp.utils import extract_keywords


class TopicShiftDetector:
    """
    Detecta cambio de tema comparando keywords
    de la nueva query con las acumuladas en el historial.
    """

    def __init__(self, threshold: float = 0.2):
        self.threshold = threshold  # porcentaje mínimo de solapamiento

    def is_shift(self, new_text: str, history_texts: List[str]) -> bool:
        new_keys: Set[str] = set(extract_keywords(new_text))
        hist_keys: Set[str] = set()
        for txt in history_texts:
            hist_keys |= set(extract_keywords(txt))
        if not hist_keys:
            return False
        overlap = new_keys & hist_keys
        # porcentaje de overlap relativo al tamaño de new_keys
        return len(overlap) / (len(new_keys) or 1) < self.threshold
