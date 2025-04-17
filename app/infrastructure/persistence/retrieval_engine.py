from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_openai import ChatOpenAI
from app.schemas.response import QueryResponse


def build_rag_chain(vector_store):
    # Prompt personalizado (en vez del hub)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Eres un asistente legal especializado en el Código de Tránsito de Bolivia. "
                "Responde de forma clara, concisa y solo con la información disponible en los documentos.",
            ),
            ("human", "{context}"),
        ]
    )

    # Inicializa un modelo de chat (ChatOpenAI suele ser la opción recomendada para prompts conversacionales)
    llm = ChatOpenAI(temperature=0.7)

    # Crea la cadena de recuperación, pasando el retriever del vector store y la cadena de combinación de documentos
    retriever = vector_store.as_retriever()
    combine_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever, combine_chain)


def process_query_with_retrieval(query: str, vector_store) -> QueryResponse:
    """
    Usa el pipeline RAG (retrieval + generación) con vector store ya cargado.
    Devuelve una respuesta estructurada con texto, fuentes y confianza.
    """
    rag_chain = build_rag_chain(vector_store)

    # Invoca la cadena con el input; el resultado puede ser un dict que incluya la respuesta en la clave "text"
    result = rag_chain.invoke({"input": query})

    # Documentos usados en la recuperación
    retrieved_docs = vector_store.as_retriever().invoke(query)
    sources = [doc.metadata.get("source", "desconocido") for doc in retrieved_docs]

    return QueryResponse(
        answer=result.get("text", "") if isinstance(result, dict) else str(result),
        sources=list(set(sources)),  # quitar duplicados
        confidence=0.85,
    )
