# app/usecases/query_interactor.py
from app.domain.repositories.idocument_retriever import IDocumentRetriever
from app.domain.repositories.iquery_expander import IQueryExpander
from app.schemas.response import QueryResponse


class QueryInteractor:
    """
    Caso de uso: expande la consulta y la envía al recuperador de documentos.
    Orquesta el flujo de negocio sin preocuparse de detalles de infraestructura.
    """

    def __init__(
        self,
        document_retriever: IDocumentRetriever,
        query_expander: IQueryExpander,
    ):
        self.document_retriever = document_retriever
        self.query_expander = query_expander

    def execute(self, query: str) -> QueryResponse:
        """
        1) Expande la consulta usando el expander inyectado.
        2) Llama al retriever inyectado para obtener la respuesta.
        3) Retorna un QueryResponse con 'answer' y 'context'.
        """
        # 1) Aplicar jerga / sinónimos
        expanded = self.query_expander.expand(query)

        # 2) Recuperar la respuesta desde FAISS / RAG
        return self.document_retriever.retrieve(expanded)
