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
    Usa un retrieval chain que primero busca documentos relevantes y luego genera una respuesta.
    """
    from langchain_openai import OpenAI
    from langchain_community.chains import RetrievalQA

    llm = OpenAI(temperature=0.7)
    retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Puedes ajustar el tipo de chain según tus necesidades
        retriever=vector_store.as_retriever(),
        return_source_documents=True,  # Opcional: para obtener también las fuentes
    )
    result = retrieval_chain.run(query)
    return result


class DocumentRetriever:
    def __init__(self, documents_directory: str):
        # Carga y divide documentos en fragmentos desde el directorio indicado
        self.documents = load_and_split_documents(documents_directory)
        self.vector_store = build_vector_store(self.documents)

    def retrieve(self, query: str) -> str:
        return process_query_with_retrieval(query, self.vector_store)
