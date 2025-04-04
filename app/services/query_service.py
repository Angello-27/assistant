import openai
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.core.config import settings

# Configurar la API key de OpenAI
openai.api_key = settings.OPENAI_API_KEY

# Definir un prompt orientado a la normativa de tránsito de Bolivia
prompt_template = """
Eres un asistente legal especializado en el Código de Tránsito de Bolivia.
Responde de forma clara y concisa a la siguiente consulta:
Pregunta: {query}
Respuesta:
"""
# Usamos PromptTemplate de langchain.prompts (si es necesario, podrías migrar a ChatPromptTemplate si usas un modelo de chat)
prompt = PromptTemplate(template=prompt_template, input_variables=["query"])

# Inicializar el modelo de lenguaje (ChatOpenAI u OpenAI, según prefieras)
llm = OpenAI(temperature=0.7)

# Crear la nueva cadena usando LCEL: componemos el prompt, el modelo y un parser que extraiga el texto
# Esto reemplaza el uso de LLMChain legacy.
chain = prompt | llm | StrOutputParser()


class QueryService:
    def __init__(self, document_retriever=None):
        """
        document_retriever: instancia de DocumentRetriever (opcional).
        Si se proporciona, se usará para la búsqueda en documentos locales.
        """
        self.document_retriever = document_retriever

    def query(self, query: str, use_retrieval: bool = True) -> str:
        """
        Procesa una consulta:
          - Si use_retrieval es True y se ha proporcionado un document_retriever,
            se utiliza la búsqueda en documentos locales.
          - De lo contrario, se procesa la consulta de forma directa usando la nueva cadena LCEL.
        """
        if use_retrieval and self.document_retriever:
            return self.document_retriever.retrieve(query)
        else:
            # Invoca la nueva cadena LCEL pasando un diccionario con el valor de "query"
            result = chain.invoke({"query": query})
            return result
