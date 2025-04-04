import os
import openai
from langchain.llms import OpenAI
from langchain.chains import LLMChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from app.core.config import settings

# Configurar la API key de OpenAI
openai.api_key = settings.OPENAI_API_KEY

# ------------------------
# Procesamiento directo sin búsqueda documental
# ------------------------

# Definir el prompt orientado a la normativa de tránsito de Bolivia
prompt_template = """
Eres un asistente legal especializado en el Código de Tránsito de Bolivia.
Responde de forma clara y concisa a la siguiente consulta:
Pregunta: {query}
Respuesta:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["query"])

# Inicializar el modelo de lenguaje y la cadena básica
llm = OpenAI(temperature=0.7)
chain = LLMChain(llm=llm, prompt=prompt)


def process_query(query: str) -> str:
    """Procesa la consulta utilizando la cadena básica de LangChain."""
    return chain.run(query=query)


# ------------------------
# Funciones de procesamiento con recuperación de documentos locales
# ------------------------


def load_documents(directory_path: str):
    """
    Carga todos los documentos de texto del directorio especificado.
    Se asume que los documentos están en formato .txt.
    """
    loader = DirectoryLoader(directory_path, glob="*.txt", loader_cls=TextLoader)
    docs = loader.load()
    return docs


def build_vector_store(documents):
    """
    Construye un vector store (FAISS) a partir de los documentos utilizando OpenAIEmbeddings.
    """
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store


def process_query_with_retrieval(query: str, vector_store) -> str:
    """
    Usa un retrieval chain que primero busca documentos relevantes y luego genera una respuesta.
    """
    retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Puedes probar con otros tipos de chain, como "map_reduce"
        retriever=vector_store.as_retriever(),
        return_source_documents=True,  # Opcional, para devolver también las fuentes
    )
    result = retrieval_chain.run(query)
    return result


# ------------------------
# Clase que encapsula la lógica de conocimiento de tránsito
# ------------------------


class TransitKnowledgeService:
    def __init__(self, documents_directory: str):
        # Carga documentos desde el directorio indicado y construye el vector store
        self.documents = load_documents(documents_directory)
        self.vector_store = build_vector_store(self.documents)

    def query(self, query: str, use_retrieval: bool = True) -> str:
        """
        Procesa una consulta:
         - Si use_retrieval es True, utiliza la búsqueda en documentos locales.
         - De lo contrario, procesa la consulta de forma directa.
        """
        if use_retrieval:
            return process_query_with_retrieval(query, self.vector_store)
        else:
            return process_query(query)
