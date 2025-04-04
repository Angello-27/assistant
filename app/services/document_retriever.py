from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS


def load_documents(directory_path: str):
    """
    Carga todos los documentos de texto del directorio especificado.
    Se asume que los documentos están en formato .txt.
    """
    loader = DirectoryLoader(directory_path, glob="*.txt", loader_cls=TextLoader)
    return loader.load()


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
    from langchain.llms import OpenAI
    from langchain.chains import RetrievalQA

    llm = OpenAI(temperature=0.7)
    retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Puedes ajustar el tipo de chain según tus necesidades
        retriever=vector_store.as_retriever(),
        return_source_documents=True,  # Opcional: para obtener también las fuentes
    )
    return retrieval_chain.run(query)


class DocumentRetriever:
    def __init__(self, directory_path: str):
        # Cargar documentos y construir el vector store
        self.documents = load_documents(directory_path)
        self.vector_store = build_vector_store(self.documents)

    def retrieve(self, query: str) -> str:
        return process_query_with_retrieval(query, self.vector_store)
