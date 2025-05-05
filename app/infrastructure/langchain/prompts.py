# app/infrastructure/langchain/prompts.py
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def get_reformulate_question_prompt() -> ChatPromptTemplate:
    """
    Prompt base para RAG: define el rol de sistema y la plantilla humana,
    y al final pide que el asistente ofrezca una recomendación de acción.
    """
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Eres un asistente legal experto en el Código de Tránsito de Bolivia. "
                "Utiliza únicamente la información proporcionada a continuación para responder a la consulta del usuario. "
                "Si no encuentras el monto exacto de la multa, pero puedes inferir si algo está prohibido o sancionado, explícalo claramente. "
                "Si no hay información útil, responde: 'No se encontró información suficiente en los documentos proporcionados.'.\n\n"
                "Al final de tu respuesta, incluye una breve sugerencia práctica de acción que el usuario pueda tomar, "
                "por ejemplo: ‘Te recomiendo que…’ o ‘Como siguiente paso, podrías…’.",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )


def get_condense_question_prompt() -> ChatPromptTemplate:
    """
    Prompt para tomar la conversación previa y un follow-up,
    y generar una pregunta autocontenida.
    """
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Eres un experto en formular preguntas claras y autocontenidas. "
                "Recibe el historial de la conversación y la siguiente entrada de usuario, "
                "y devuelve únicamente la pregunta reformulada, sin añadir nada más.",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )
