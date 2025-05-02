# app/infrastructure/langchain/retrieval_engine.py
from typing import List
import logging
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from app.domain.repositories.irag_engine import IRetrievalEngine
from app.schemas.response import QueryResponse, Document as APIDocument
from .prompts import get_rag_chat_prompt


class RagEngine(IRetrievalEngine):
    """
    Implementación de IRetrievalEngine que:
     - Mantiene memoria conversacional resumida (ConversationSummaryBufferMemory)
     - Usa ConversationalRetrievalChain para inyectar contexto
     - Optimiza el historial guardando solo resúmenes de turns anteriores
    """

    def __init__(self, vector_store):
        # Logger para capturar errores internos
        self.logger = logging.getLogger(__name__)

        # 1) LLM compartido para generación y resumen
        self.llm = ChatOpenAI(temperature=0.7)

        # 2) Memoria de conversación resumida
        #    Guarda un breve resumen de cada intercambio, no el texto completo
        self.memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            memory_key="chat_history",  # variable usada en el prompt
            input_key="question",  # nombre de la entrada
            output_key="answer",  # nombre de la salida
            max_token_limit=512,  # tope para tamaño del resumen
        )

        # 3) Prompt base con recomendación final
        prompt = get_rag_chat_prompt()
        # 4) Cadena para combinar documentos y el prompt
        self.combine_chain = create_stuff_documents_chain(self.llm, prompt)

        # 5) Construir la ConversationalRetrievalChain
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=vector_store.as_retriever(),
            memory=self.memory,
            combine_docs_chain=self.combine_chain,
        )

    def process_with_context(self, query: str) -> QueryResponse:
        """
        1) Ejecuta la ConversationalRetrievalChain con la query.
        2) Actualiza la memoria con un resumen del turno.
        3) Formatea los documentos fuente como snippets.
        4) Devuelve QueryResponse(answer, context).
        """
        try:
            result = self.chain({"question": query})
        except Exception as e:
            self.logger.error(
                "Error en ConversationalRetrievalChain: %s", e, exc_info=True
            )
            raise RuntimeError("Fallo en el motor de recuperación conversacional.")

        # Extraer la respuesta generada
        answer: str = result.get("answer", "")

        # Recuperar documentos del turno actual (si existen)
        docs = result.get("retrieved_documents", [])

        # Formatear top-3 documentos como snippets
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

    def retrieve(self, query: str) -> QueryResponse:
        """
        Método del contrato IRetrievalEngine: delega a process_with_context().
        """
        return self.process_with_context(query)
