from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents(directory_path: str):
    """
    Carga todos los documentos de texto del directorio especificado.
    Se asume que los documentos están en formato .txt.
    """
    loader = DirectoryLoader(
        directory_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    return loader.load()


def load_and_split_documents(directory_path: str):
    """
    Carga y divide en fragmentos los documentos de texto del directorio especificado.
    """
    docs = load_documents(directory_path)
    # Configura el text splitter: chunk_size es el tamaño máximo del fragmento,
    # chunk_overlap es la cantidad de caracteres que se solapan entre fragmentos.
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)
    return split_docs


def build_vector_store(documents):
    """
    Construye un vector store (FAISS) a partir de los documentos utilizando OpenAIEmbeddings.
    """
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(documents, embeddings)


def process_query_with_retrieval(query: str, vector_store) -> str:
    """
    Usa el nuevo enfoque LCEL para buscar documentos relevantes en el vector store y
    generar una respuesta a partir de ellos.
    """
    from langchain import hub
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.chains import create_retrieval_chain
    from langchain_openai import ChatOpenAI

    # Extrae un prompt de ejemplo desde hub (puedes personalizarlo o definir el tuyo propio)
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    # Inicializa un modelo de chat (ChatOpenAI suele ser la opción recomendada para prompts conversacionales)
    llm = ChatOpenAI(temperature=0.7)

    # Crea una cadena para combinar documentos usando el prompt obtenido
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

    # Crea la cadena de recuperación, pasando el retriever del vector store y la cadena de combinación de documentos
    rag_chain = create_retrieval_chain(vector_store.as_retriever(), combine_docs_chain)

    # Invoca la cadena con el input; el resultado puede ser un dict que incluya la respuesta en la clave "text"
    result = rag_chain.invoke({"input": query})

    # Si el resultado es un diccionario y contiene la clave "text", se extrae esa respuesta
    if isinstance(result, dict) and "text" in result:
        return result["text"]
    return result


class DocumentRetriever:
    def __init__(self, documents_directory: str):
        # Carga y divide documentos en fragmentos desde el directorio indicado
        self.documents = load_and_split_documents(documents_directory)
        self.vector_store = build_vector_store(self.documents)

    def retrieve(self, query: str) -> str:
        return process_query_with_retrieval(query, self.vector_store)
