from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_openai import ChatOpenAI
from app.schemas.response import QueryResponse


def build_rag_chain(vector_store):
    # Prompt con variable 'context' porque es lo que recibe el LLM luego del formato
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Eres un asistente legal experto en el Código de Tránsito Boliviano. "
                "Utiliza únicamente la información proporcionada a continuación para responder. "
                "Si no encuentras información suficiente, responde exactamente: "
                "'No se encontró información suficiente en los documentos proporcionados.'",
            ),
            (
                "human",
                "{context}",
            ),  # Esta variable es llenada por LangChain con los documentos recuperados
        ]
    )

    # LLM configurado
    llm = ChatOpenAI(temperature=0.7)

    # Crea la cadena que combina documentos y los convierte en texto
    combine_chain = create_stuff_documents_chain(llm, prompt)

    # Crea el retriever con el vector store
    retriever = vector_store.as_retriever()

    # Une el retriever con la cadena de combinación (RAG)
    return create_retrieval_chain(retriever, combine_chain)


def process_query_with_retrieval(query: str, vector_store) -> QueryResponse:
    """
    Ejecuta el pipeline RAG completo: recuperación + generación.
    """
    print("🔍 [RAG] Procesando consulta:", query)

    # Construye el pipeline RAG
    rag_chain = build_rag_chain(vector_store)

    # Ejecuta el pipeline pasando la consulta como 'input'
    result = rag_chain.invoke(
        {"input": query}
    )  # ¡IMPORTANTE! 'input' es la entrada esperada

    # Extrae documentos usados
    retrieved_docs = vector_store.as_retriever().invoke(query)
    sources = [doc.metadata.get("source", "desconocido") for doc in retrieved_docs]

    # Debug: imprime el contexto enviado al LLM
    context = "\n\n".join(
        [
            f"Fuente: {doc.metadata.get('source', 'desconocido')}\n{doc.page_content}"
            for doc in retrieved_docs
        ]
    )
    print("🧾 Contexto completo enviado al LLM:\n", context)

    # Devuelve la respuesta estructurada
    return QueryResponse(
        answer=result.get("answer", "") if isinstance(result, dict) else str(result),
        sources=list(set(sources)),
        confidence=0.85,
    )
