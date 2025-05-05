# app/infrastructure/langchain/prompts.py
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def get_reformulate_question_prompt() -> ChatPromptTemplate:
    """
    Prompt base para RAG: define el rol de sistema, recibe los documentos
    recuperados como texto en {context} y la pregunta del usuario {input}.
    Incluye sugerencia de acción al final.
    """

    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Eres un asistente legal experto en el Código de Tránsito de Bolivia. "
                "Utiliza únicamente la información proporcionada a continuación para responder a la consulta del usuario. "
                "Si no encuentras el monto exacto de la multa, pero puedes inferir si algo está prohibido o sancionado, explícalo claramente. "
                "Si no hay información útil, responde: 'No se encontró información suficiente en los documentos proporcionados.'. "
                "Al final de tu respuesta, incluye una breve sugerencia práctica de acción que el usuario pueda tomar.",
            ),
            # Aquí inyectamos el texto de contextos recuperados:
            ("system", "{context}"),
            # Luego el historial de la conversación:
            MessagesPlaceholder(variable_name="chat_history"),
            # Finalmente la pregunta actual:
            ("human", "{input}"),
        ]
    )


def get_condense_question_prompt() -> ChatPromptTemplate:
    """
    Prompt para formular preguntas autocontenidas.
    Recibe historial como {chat_history} y la entrada del usuario {input}.
    """
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Eres un asistente legal experto en el Código de Tránsito de Bolivia. "
                "Recibe el historial de SOLO las **preguntas** anteriores "
                "y la siguiente entrada del usuario. Si la nueva entrada es ambigua, "
                "reformularla de modo que siga hablando del mismo tema del historial "
                "Devuelve únicamente la pregunta autocontenida.",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )
