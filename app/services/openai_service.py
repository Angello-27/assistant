import openai
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from app.core.config import settings

# Configurar la API key de OpenAI
openai.api_key = settings.OPENAI_API_KEY

# Definir el prompt orientado a la normativa de tránsito de Bolivia
prompt_template = """
Eres un asistente legal especializado en el Código de Tránsito de Bolivia.
Responde de forma clara y concisa a la siguiente consulta:
Pregunta: {query}
Respuesta:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["query"])

# Inicializar el modelo de lenguaje
llm = OpenAI(temperature=0.7)
chain = LLMChain(llm=llm, prompt=prompt)


def process_query(query: str) -> str:
    """Procesa la consulta usando LangChain y devuelve la respuesta generada."""
    return chain.run(query=query)
