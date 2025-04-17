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

    def __init__(self):
        self.embeddings = OpenAIEmbeddings()

    def build_vectorstore(self, fragments: List[Fragment]) -> FAISS:
        docs = []

        for frag in fragments:
            metadata = {
                "source": frag.source,
                "id": frag.id,
                "position": frag.position,
                "tags": frag.tags,
                "synonyms": frag.synonyms,
            }
            docs.append(Document(page_content=frag.content, metadata=metadata))

        return FAISS.from_documents(docs, self.embeddings)
