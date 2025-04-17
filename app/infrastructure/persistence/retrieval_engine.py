from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_openai import ChatOpenAI


def process_query_with_retrieval(query: str, vector_store) -> str:
    """
    Usa el nuevo enfoque LCEL para buscar documentos relevantes en el vector store y
    generar una respuesta a partir de ellos.
    Devuelve respuesta estructurada con texto, fuentes y confianza.
    """

    # Extrae un prompt de ejemplo desde hub (puedes personalizarlo o definir el tuyo propio)
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    # Inicializa un modelo de chat (ChatOpenAI suele ser la opción recomendada para prompts conversacionales)
    llm = ChatOpenAI(temperature=0.7)
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

    # Crea la cadena de recuperación, pasando el retriever del vector store y la cadena de combinación de documentos
    retriever = vector_store.as_retriever()
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

    # Invoca la cadena con el input; el resultado puede ser un dict que incluya la respuesta en la clave "text"
    result = rag_chain.invoke({"input": query})

    # Documentos usados en la recuperación
    retrieved_docs = retriever.get_relevant_documents(query)
    sources = [doc.metadata.get("source", "desconocido") for doc in retrieved_docs]

    # Si el resultado es un diccionario y contiene la clave "text", se extrae esa respuesta
    return {
        "answer": result.get("text", "") if isinstance(result, dict) else str(result),
        "sources": list(set(sources)),  # quitar duplicados
        "confidence": 0.85,  # valor fijo, o podrías calcular algo más sofisticado
    }
