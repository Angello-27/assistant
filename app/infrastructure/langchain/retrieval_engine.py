# app/infrastructure/langchain/retrieval_engine.py
import logging
from typing import List, Any

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from app.domain.repositories.irag_engine import IRetrievalEngine
from app.schemas.response import QueryResponse, Document as APIDocument
from app.infrastructure.langchain.prompts import (
    get_condense_question_prompt,
    get_reformulate_question_prompt,
)

logger = logging.getLogger(__name__)


class RagEngine(IRetrievalEngine):
    """
    Motor RAG basado en LangChain LCEL.
    Combina:
      - Reformulación de preguntas (follow-up → autocontenida).
      - Recuperación de documentos (vector store FAISS).
      - Generación de respuesta con contexto legal.
      - Gestión del historial de la conversación completa.
    """

    def __init__(self, vector_store):
        # 1) Inicializar LLM: ChatOpenAI de la comunidad
        self.llm = ChatOpenAI(temperature=0.7)

        # 2) Configurar memoria de conversación (almacena mensajes humanos y respuestas)
        reformulate_prompt = get_condense_question_prompt()
        self.history_aware_retriever = create_history_aware_retriever(
            llm=self.llm,
            retriever=vector_store.as_retriever(),
            prompt=reformulate_prompt,
        )

        # 3) Cadena de combinación "stuff" con contexto
        qa_prompt = get_reformulate_question_prompt()
        self.stuff_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=qa_prompt,
            document_variable_name="context",
        )

        # 4) Cadena completa RAG: recuperación + combinación
        self.chain = create_retrieval_chain(
            retriever=self.history_aware_retriever,
            combine_docs_chain=self.stuff_chain,
        )

        # 5) Vector store para consultas directas
        self.vector_store = vector_store

        # 6) Historial de conversación: lista de HumanMessage y AIMessage
        self.chat_history: List[Any] = []

    def retrieve(self, query: str) -> QueryResponse:
        """
        Ejecuta el pipeline RAG completo:
          1) Envía el historial y la consulta a la cadena.
          2) Obtiene la respuesta generada.
          3) Recupera manualmente los 3 documentos más relevantes mediante invocación.
          4) Guarda tanto la entrada como la respuesta en el historial.
          5) Devuelve QueryResponse.
        """
        try:
            # Ejecutar la cadena conversacional: reformulación + recuperación + generación
            result = self.chain.invoke(
                {
                    "input": query,
                    "chat_history": self.chat_history,
                }
            )
        except Exception as e:
            logger.error("Error en pipeline RAG: %s", e, exc_info=True)
            raise RuntimeError("Fallo en el motor RAG.") from e

        # La clave de la respuesta puede variar entre 'answer' o 'result'
        answer = result.get("answer") or result.get("result") or ""

        # Extraer top-3 documentos usando invoke sobre el runnable
        retriever = self.vector_store.as_retriever()
        docs = retriever.invoke(query)

        context: List[APIDocument] = []
        for doc in docs[:3]:
            context.append(
                APIDocument(
                    id=doc.metadata.get("id", "sin_id"),
                    metadata={
                        "source": doc.metadata.get("source"),
                        "tags": doc.metadata.get("tags", []),
                    },
                    page_content=doc.page_content[:300].strip() + "...",
                    type="document",
                )
            )

        # Guardar en historial para la siguiente interacción
        self.chat_history.append(HumanMessage(content=query))
        self.chat_history.append(AIMessage(content=answer))

        # Devolver respuesta y contexto
        return QueryResponse(answer=answer, context=context)
