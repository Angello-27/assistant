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
    Construye la cadena RAG una sola vez al inicializar.
    """

    def __init__(self, vector_store):
        # 1) Preparar prompt
        prompt = get_rag_chat_prompt()
        # 2) Instanciar LLM
        llm = ChatOpenAI(temperature=0.7)
        # 3) Crear sub-chain de combinación de documentos
        combine_chain = create_stuff_documents_chain(llm, prompt)
        # 4) Crear retriever sobre vector_store
        retriever = vector_store.as_retriever()
        # 5) Armar cadena RAG completa
        self.chain = create_retrieval_chain(retriever, combine_chain)
        self.vector_store = vector_store

    def retrieve(self, query: str) -> QueryResponse:
        """
        Ejecuta la cadena RAG con la consulta dada.
        Devuelve QueryResponse con texto y contexto (artículos fuente).
        """
        # Ejecutar pipeline RAG
        result = self.chain.invoke({"input": query})
        # Recuperar documentos para el contexto
        docs = self.vector_store.as_retriever().invoke(query)
        context = [
            APIDocument(
                id=doc.metadata["id"],
                metadata=doc.metadata,
                page_content=doc.page_content,
                type="document",
            )
            for doc in docs
        ]
        return QueryResponse(
            answer=(
                result.get("answer", "") if isinstance(result, dict) else str(result)
            ),
            context=context,
        )
