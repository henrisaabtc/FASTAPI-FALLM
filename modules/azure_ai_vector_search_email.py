"""Handle all azure vector search"""
from typing import Tuple

from langchain.docstore.document import Document

from azure.search.documents.aio import SearchClient

from azure.search.documents.models import (
    QueryType,
    QueryCaptionType,
    QueryAnswerType,
    VectorizedQuery,
)
from azure.core.credentials import AzureKeyCredential

from config.config import EnvParam

from modules.embeddings import Embeddings

credential = AzureKeyCredential(EnvParam.AZURE_AI_SEARCH_KEY)

search_client_mail = SearchClient(
    endpoint=EnvParam.AZURE_AI_SEARCH_ENDPOINT,
    index_name=EnvParam.AZURE_AI_SEARCH_INDEX_NAME_MAIL,
    credential=credential,
    logging_enable=False,
)


embeddings = Embeddings()


class AzureAIVectorSearchEmail:
    """Handle all azure search query"""

    async def get_chunks(self, list_queries: list[str]) -> list[Tuple[Document, float]]:
        """Function that get chunk and create string context for semantic"""

        chunks: list[Tuple[Document, float]] = []

        for query in list_queries:
            embedding = await embeddings.embedding_query(query=query)

            vector_query = VectorizedQuery(
                vector=embedding,
                k_nearest_neighbors=3,
                fields="vector",
                exhaustive=True,
            )

            results = await search_client_mail.search(
                search_text=query,
                vector_queries=[vector_query],
                select=[
                    "subject",
                    "chunk",
                    "sender",
                    "cced",
                    "bcced",
                    "has_attachment",
                    "date_sent",
                ],
                query_type=QueryType.SEMANTIC,
                semantic_configuration_name=f"{EnvParam.AZURE_AI_SEARCH_INDEX_NAME_MAIL}_semantic_config",
                query_caption=QueryCaptionType.EXTRACTIVE,
                query_answer=QueryAnswerType.EXTRACTIVE,
                top=3,
            )

            async for result in results:
                chunk = Document(
                    page_content=result["chunk"],
                    metadata={
                        "file_name": result.get("subject"),
                        "sender": result.get("sender"),
                        "cced": result.get("cced"),
                        "bcced": result.get("bcced"),
                        "has_attachment": result.get("has_attachment"),
                        "date_sent": result.get("date_sent"),
                    },
                )

                chunks.append((chunk, result["@search.reranker_score"]))

        return chunks
