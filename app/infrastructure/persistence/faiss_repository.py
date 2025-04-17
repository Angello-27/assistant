import os
from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from app.domain.entities.fragment import Fragment


class FAISSRepository:
    """
    Clase responsable de almacenar y consultar documentos en FAISS
    a partir de los documentos utilizando OpenAIEmbeddings.
    """

    def __init__(self, persist_path: str = "faiss_index"):
        self.persist_path = persist_path
        self.embeddings = OpenAIEmbeddings()

    def build_vectorstore(self, fragments: List[Fragment]) -> FAISS:
        docs = [
            Document(
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

        vectorstore = FAISS.from_documents(docs, self.embeddings)
        vectorstore.save_local(self.persist_path)
        return vectorstore

    def load_vectorstore(self) -> FAISS:
        if not os.path.exists(self.persist_path):
            raise FileNotFoundError(
                "FAISS index no encontrado. Debes generarlo primero."
            )
        return FAISS.load_local(self.persist_path, self.embeddings)
