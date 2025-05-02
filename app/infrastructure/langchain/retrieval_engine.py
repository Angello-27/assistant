# app/infrastructure/langchain/retrieval_engine.py
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_openai import ChatOpenAI
from app.domain.repositories.irag_engine import IRetrievalEngine
from app.schemas.response import QueryResponse, Document as APIDocument
from .prompts import get_rag_chat_prompt


class RagEngine(IRetrievalEngine):
    """
    Implementación de IRetrievalEngine usando LangChain + OpenAI + FAISS.
    """

    def __init__(self, vector_store):
        # Construye la cadena RAG una sola vez
        prompt = get_rag_chat_prompt()
        # LLM configurado
        llm = ChatOpenAI(temperature=0.7)

        # Crea la cadena que combina documentos y los convierte en texto
        combine = create_stuff_documents_chain(llm, prompt)
        # Crea el retriever con el vector store
        retriever = vector_store.as_retriever()

        # Une el retriever con la cadena de combinación (RAG)
        self.chain = create_retrieval_chain(retriever, combine)
        self.vector_store = vector_store

    def retrieve(self, query: str) -> QueryResponse:
        """
        Usa el pipeline RAG (retrieval + generación) con vector store ya cargado.
        Devuelve una respuesta estructurada con texto y artículos fuente usados.
        """
        # 1) Generar respuesta
        result = self.chain.invoke({"input": query})
        # 2) Recolectar fragmentos usados
        docs = self.vector_store.as_retriever().invoke(query)
        documents_context = [
            APIDocument(
                id=doc.metadata["id"],
                metadata=doc.metadata,
                page_content=doc.page_content,
                type="document",
            )
            for doc in docs
        ]
        # 3) Devolver QueryResponse
        return QueryResponse(
            answer=(
                result.get("answer", "") if isinstance(result, dict) else str(result)
            ),
            context=documents_context,
        )
