from app.infrastructure.persistence.document_loader import load_documents_from_directory
from app.infrastructure.persistence.faiss_repository import FAISSRepository
from app.infrastructure.persistence.retrieval_engine import process_query_with_retrieval
from app.infrastructure.nlp.splitter import split_document_spacy


class DocumentRetriever:
    """
    Orquesta el flujo: cargar documentos → fragmentar → vectorizar → consultar.
    Permite seleccionar la estrategia de fragmentación.
    """

    def __init__(self, documents_directory: str, split_mode: str = "articles"):
        print("📦 [Retriever] Inicializando DocumentRetriever...")
        self.documents_directory = documents_directory
        self.repo = FAISSRepository()  # repo ahora está disponible siempre

        try:
            self.vector_store = self.repo.load_vectorstore()
            print("✅ [Retriever] Vector store cargado desde disco.")
        except FileNotFoundError:
            print("⚠️ [Retriever] Vector store no encontrado, reconstruyendo...")
            self.vector_store = self._prepare_vectorstore()

    def _prepare_vectorstore(self):
        print("⚙️ [Retriever] Fragmentando y vectorizando documentos...")
        # Carga y divide documentos en fragmentos desde el directorio indicado
        docs = load_documents_from_directory(self.documents_directory)

        fragments = []
        for doc in docs:
            content = doc.page_content
            filename = doc.metadata.get("source", "documento")
            print(f"📂 Cargando documento: {filename}")
            fragments.extend(split_document_spacy(content, filename))
        print(f"✅ [Retriever] Total de fragmentos generados: {len(fragments)}")

        repo = FAISSRepository()
        return repo.build_vectorstore(fragments)

    def retrieve(self, query: str) -> str:
        return process_query_with_retrieval(query, self.vector_store)
