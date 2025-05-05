# app/infrastructure/langchain/retrieval_engine.py
import logging
from typing import List

from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.language_models.base import BaseLanguageModel

from app.domain.repositories.irag_engine import IRetrievalEngine
from app.schemas.response import QueryResponse, Document as APIDocument
from app.infrastructure.langchain.prompts import (
    get_condense_question_prompt,
    get_reformulate_question_prompt,
)

logger = logging.getLogger(__name__)


class RagEngine(IRetrievalEngine):
    """
    Motor RAG basado en LangChain.
    Combina:
      - Reformulación de preguntas (follow-up → autocontenida).
      - Recuperación de documentos (vector store FAISS).
      - Generación de respuesta con contexto legal.
      - Memoria de la conversación completa.
    """

    def __init__(self, vector_store):
        # 1) Inicializar LLM: ChatOpenAI de la comunidad
        self.llm: BaseLanguageModel = ChatOpenAI(temperature=0.7)

        # 2) Configurar memoria de conversación (almacena mensajes humanos y respuestas)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

        # 3) Construir la cadena conversacional RAG
        #    - question_generator_chain_kwargs: prompt para reformular follow-up
        #    - combine_docs_chain_kwargs: prompt para generación final con contexto legal
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=vector_store.as_retriever(),
            memory=self.memory,
            question_generator_chain_kwargs={"prompt": get_condense_question_prompt()},
            combine_docs_chain_kwargs={"prompt": get_reformulate_question_prompt()},
        )

        # Guardar referencia al vector_store para obtener top-k manualmente
        self.vector_store = vector_store

    def retrieve(self, query: str) -> QueryResponse:
        """
        Ejecuta el pipeline RAG completo:
          1) Reformula la pregunta usando el prompt de condensación.
          2) Recupera documentos relevantes mediante FAISS.
          3) Genera la respuesta con contexto y sugerencia práctica.
          4) Devuelve un QueryResponse con answer y contexto de top-3.
        """
        try:
            # Ejecutar la cadena conversacional: reformulación + recuperación + generación
            result = self.chain({"question": query})
        except Exception as e:
            logger.error("Error en pipeline RAG: %s", e, exc_info=True)
            raise RuntimeError("Fallo en el motor RAG.") from e

        # Obtener la respuesta generada
        answer = result.get("answer", "").strip()

        # 4) Extraer manualmente los top-3 documentos para el contexto
        docs = self.vector_store.as_retriever().get_relevant_documents(query)
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

        return QueryResponse(answer=answer, context=context)
