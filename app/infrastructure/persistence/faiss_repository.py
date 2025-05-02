# app/infrastructure/persistence/faiss_repository.py
import os
from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document as LCDocument
from app.domain.entities.fragment import Fragment
from app.domain.repositories.ivector_repository import IVectorRepository


class FAISSRepository(IVectorRepository):
    """
    Implementación de IVectorRepository usando FAISS en disco.
    """

    def __init__(self, persist_path: str = "faiss_index"):
        self.persist_path = persist_path
        self.embeddings = OpenAIEmbeddings()

    def build(self, fragments: List[Fragment]) -> FAISS:
        """
        Construye un FAISS index a partir de la lista de Fragment,
        lo persiste en disco y lo retorna.
        """
        docs = [
            LCDocument(
                page_content=frag.content,
                metadata={
                    "source": frag.source,
                    "id": frag.id,
                    "position": frag.position,
                    "tags": frag.tags,
                    "synonyms": frag.synonyms,
                },
            )
            for frag in fragments
        ]
        index = FAISS.from_documents(docs, self.embeddings)
        print("[FAISS] Guardando índice FAISS en disco...")
        index.save_local(self.persist_path)
        print(f"[FAISS] Vector store guardado en: {self.persist_path}")
        return index

    def load(self) -> FAISS:
        """
        Carga el FAISS index desde disco.
        """
        if not os.path.exists(self.persist_path):
            raise FileNotFoundError(
                f"FAISS index no encontrado en {self.persist_path}. Genera el índice primero."
            )
        print("[FAISS] Cargando índice FAISS desde disco...")
        return FAISS.load_local(
            self.persist_path,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
