"""Handle all azure vector search"""
from typing import Tuple
from xml.etree import ElementTree

import requests

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

from modules import logger

from modules.embeddings import Embeddings

credential = AzureKeyCredential(EnvParam.AZURE_AI_SEARCH_KEY)

search_client = SearchClient(
    endpoint=EnvParam.AZURE_AI_SEARCH_ENDPOINT,
    index_name=EnvParam.AZURE_AI_SEARCH_INDEX_NAME_SHAREPOINT,
    credential=credential,
    logging_enable=False,
)

embeddings = Embeddings()


class AzureAIVectorSearch:
    """Handle all azure search query"""

    async def get_all_docs_str(self) -> str:
        """Get all docs name in share point"""

        documents_name: str = ""

        try:
            response = requests.get(EnvParam.BLOB_ENDPOINT, timeout=5000)

            response.raise_for_status()

            root = ElementTree.fromstring(response.content)

            names = [blob.find("Name").text for blob in root.findall(".//Blob")]

            documents_name_list: list[str] = []

            for name in names:
                logger.info("All docs %s", str(response))

                if name not in documents_name_list:
                    documents_name += f"- {name}\n"

                    documents_name_list.append(name)

        except Exception as e:
            logger.error("Error getting all docs %s", e)

            documents_name = ""

        logger.warning(documents_name)

        logger.warning("Nb documents %s", len(documents_name_list))

        return documents_name

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

            results = await search_client.search(
                search_text=query,
                vector_queries=[vector_query],
                select=["title", "chunk", "source_url"],
                query_type=QueryType.SEMANTIC,
                semantic_configuration_name=f"{EnvParam.AZURE_AI_SEARCH_INDEX_NAME_SHAREPOINT}_semantic_config",
                query_caption=QueryCaptionType.EXTRACTIVE,
                query_answer=QueryAnswerType.EXTRACTIVE,
                top=3,
            )

            async for result in results:
                chunk = Document(
                    page_content=result["chunk"],
                    metadata={
                        "file_name": result["title"],
                        "source_url": result["source_url"],
                    },
                )

                chunks.append((chunk, result["@search.reranker_score"]))

        return chunks
