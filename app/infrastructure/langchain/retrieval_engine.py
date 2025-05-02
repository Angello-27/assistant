# app/infrastructure/langchain/retrieval_engine.py
import logging
from typing import List

from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain, create_history_aware_retriever, RetrievalQA
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from app.domain.repositories.irag_engine import IRetrievalEngine
from app.schemas.response import QueryResponse, Document as APIDocument
from .prompts import get_rag_chat_prompt

logger = logging.getLogger(__name__)


class RagEngine(IRetrievalEngine):
    """
    Implementación de IRetrievalEngine con:
      - Historial completo (ConversationBufferMemory)
      - Reformulación de preguntas sueltas
      - QA tipo 'stuff' con sugerencia final
      - Resúmenes automáticos cuando la conversación excede N turnos
    """

    # Número de turnos tras los que auto-resumimos
    SUMMARY_THRESHOLD = 5

    def __init__(self, vector_store):
        # 1) LLM compartido
        self.llm = ChatOpenAI(temperature=0.7)

        # 2) Memoria básica que guarda lista de mensajes
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
        )

        # 3) Cadena LLMChain para resumir la historia
        summary_system = (
            "Resume brevemente la siguiente historia de chat en una frase "
            "conservando los detalles legales más relevantes."
        )
        summary_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", summary_system),
                MessagesPlaceholder(variable_name="chat_history"),
            ]
        )
        self.summary_chain = LLMChain(llm=self.llm, prompt=summary_prompt)

        # 4) Prompt y retriever “history-aware” para reformular preguntas sueltas
        reformulate_system = (
            "Dada la conversación previa y la última pregunta, "
            "reformula la consulta para que sea autocontenida. "
            "No respondas aún, solo reformula."
        )
        reformulate_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", reformulate_system),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )
        base_retriever = vector_store.as_retriever()
        self.history_aware_retriever = create_history_aware_retriever(
            llm=self.llm,
            retriever=base_retriever,
            prompt=reformulate_prompt,
        )

        # 5) Prompt final RAG con recomendación práctica
        rag_prompt = get_rag_chat_prompt()

        # 6) Cadena RetrievalQA usando tu prompt, sin source docs automáticos
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.history_aware_retriever,
            return_source_documents=False,
            chain_type_kwargs={"prompt": rag_prompt},
        )

        # guardamos vector_store para extraer fuentes manualmente
        self.vector_store = vector_store

    def retrieve(self, query: str) -> QueryResponse:
        """
        1) (Opcional) Si la conversación supera SUMMARY_THRESHOLD turnos,
           generamos un resumen y lo sustituimos por la historia completa.
        2) Reformulamos + ejecutamos QA.
        3) Recuperamos top-3 docs manualmente y devolvemos QueryResponse.
        """
        # 1a) Cargar historia actual
        mem_vars = self.memory.load_memory_variables({})
        history = mem_vars.get("chat_history", [])

        # 1b) Si excede umbral, resumir y reiniciar historia
        if len(history) >= self.SUMMARY_THRESHOLD:
            try:
                summary = self.summary_chain.run({"chat_history": history})
                # Volvemos a empezar la memoria sólo con un mensaje de “Resumen”
                self.memory = ConversationBufferMemory(
                    memory_key="chat_history", return_messages=True
                )
                self.memory.save_context(
                    {"human_input": "<resumen histórico>"},
                    {"generated_text": summary},
                )
            except Exception as e:
                logger.warning("Error al generar resumen de memoria: %s", e)

        # 2) Ejecutar QA con reformulación automática
        try:
            qa_input = {
                "query": query,
                "chat_history": self.memory.load_memory_variables({})["chat_history"],
            }
            result = self.qa_chain(qa_input)
        except Exception as e:
            logger.error("Error en QA chain: %s", e, exc_info=True)
            raise RuntimeError("Fallo en el motor RAG.")

        answer = result.get("result") or result.get("answer", "")

        # 3) Recuperar manualmente top-3 documentos como contexto
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

        # 4) Guardar en memoria este turno (humano + asistente)
        self.memory.save_context(
            {"human_input": query},
            {"generated_text": answer},
        )

        return QueryResponse(answer=answer, context=context)
