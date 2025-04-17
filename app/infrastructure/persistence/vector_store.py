from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from app.domain.entities.fragment import Fragment


def store_in_vector_db(fragments: List[Fragment]):
    embeddings = OpenAIEmbeddings()
    docs = []

    for frag in fragments:
        metadata = {
            "source": frag.source,
            "id": frag.id,
            "position": frag.position,
            "tags": frag.tags,
            "synonyms": frag.synonyms
        }
        docs.append(Document(page_content=frag.content, metadata=metadata))

    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore
