"""Module that get context from embeddings db"""

import re
import time
from typing import Optional, Tuple

from pydantic import BaseModel, ConfigDict

from langchain.schema import Document

from config.config import EnvParam

from modules import logger
from modules.queries import Queries
from modules.embeddings import Embeddings
from modules.doc_local_search import DocLocalSearch
from modules.azure_ai_vector_search import AzureAIVectorSearch
from modules.azure_ai_vector_search_email import AzureAIVectorSearchEmail
from modules.azure_ai_vector_search_glpi import AzureAIVectorSearchGLPI

from modules.google import Google


if EnvParam.USE_AZURE:
    if EnvParam.EMBEDDING_MODEL_AZURE in {"text-embedding-ada-002"}:
        CHUNK_SCORE_LIMIT: float = EnvParam.CHUNK_SCORE_LIMIT_EMBEDDINGS_ADA

    else:
        CHUNK_SCORE_LIMIT = EnvParam.CHUNK_SCORE_LIMIT_EMBEDDINGS_3

else:
    if EnvParam.EMBEDDINGS_MODEL_OPEN_AI in {"text-embedding-ada-002"}:
        CHUNK_SCORE_LIMIT = EnvParam.CHUNK_SCORE_LIMIT_EMBEDDINGS_ADA

    else:
        CHUNK_SCORE_LIMIT = EnvParam.CHUNK_SCORE_LIMIT_EMBEDDINGS_3

embeddings = Embeddings()


class ContextChunk(BaseModel):
    """One chunk for context"""

    index: int

    content: str

    similarity: float

    is_used_in_answer: bool = False

    document: str

    url: Optional[str] = None

    sender: Optional[str] = None

    cced: Optional[str] = None

    bcced: Optional[str] = None

    has_attachment: Optional[str] = None

    date_sent: Optional[str] = None


class Context(BaseModel):
    """Class that get context from embeddings db"""

    context_list: list[ContextChunk] = []

    doc_used: list[str] = []

    context: str = ""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    user_input: str

    memory_list: Optional[list[dict[str, str]]]

    chunk_retreiver: Optional[
        DocLocalSearch
        | AzureAIVectorSearch
        | Google
        | AzureAIVectorSearchEmail
        | AzureAIVectorSearchGLPI
    ]

    queries: Optional[Queries] = None

    async def get_context(self) -> None:
        """Function that create the db and the get all context chunks"""

        if not self.chunk_retreiver:
            return

        if isinstance(self.chunk_retreiver, DocLocalSearch):
            documents_names: str = self.chunk_retreiver.documents.documents_names

        elif isinstance(self.chunk_retreiver, Google):
            documents_names = "Google search"

        elif isinstance(self.chunk_retreiver, AzureAIVectorSearchEmail):
            documents_names = "Email"

        elif isinstance(self.chunk_retreiver, AzureAIVectorSearchGLPI):
            documents_names = "GLPI"

        else:
            documents_names = await self.chunk_retreiver.get_all_docs_str()

        logger.info("Documents names %s", documents_names)

        self.queries = Queries(
            user_input=self.user_input,
            memory_list=self.memory_list,
            documents_names=documents_names,
            is_google=isinstance(self.chunk_retreiver, Google),
        )

        await self.queries.get_queries()

        start_context_time = time.time()

        if len(self.queries.all_queries) > 0:
            chunks: list[
                Tuple[Document, float]
            ] = await self.chunk_retreiver.get_chunks(
                list_queries=self.queries.all_queries
            )

            self.select_chunk(docs=chunks)

            try:
                self.create_str_context()

            except Exception as e:
                logger.error("Error creating context %s", e)

                self.context = ""

                self.context_list = []

            if len(self.context) == 0:
                logger.warning("There is no context with these queries")

        else:
            logger.warning("There is no queries")

        logger.warning("Get chunk time %f", time.time() - start_context_time)

    def add_context_chunk_to_source(self, index_chunk: int) -> None:
        """Add context to source (change bool val)"""

        for chunk in self.context_list:
            if chunk.index == index_chunk and not chunk.is_used_in_answer:
                chunk.is_used_in_answer = True

                logger.info("Add chunk %d to source", index_chunk)

    def create_str_context(self) -> None:
        """Function that create string context from chunk"""

        context: str = ""

        context_by_doc: dict = {}

        for doc in self.doc_used:
            context_by_doc[doc] = {"content": "", "score": 0.0}

            for chunk in self.context_list:
                if chunk.document == doc:
                    context_by_doc[doc]["content"] = (
                        context_by_doc[doc]["content"] + "\n\n" + chunk.content
                    )

                    if chunk.similarity > context_by_doc[doc]["score"]:
                        context_by_doc[doc]["score"] = chunk.similarity

        context_list = [
            {"doc": doc, "content": info["content"], "score": info["score"]}
            for doc, info in context_by_doc.items()
        ]

        sorted_context_list = sorted(
            context_list, key=lambda x: x["score"], reverse=True
        )

        for chunk in sorted_context_list:
            try:
                chunk_content: str = chunk["content"]

                pattern = r"[^\x20-\x7E\n\r]+"

                chunk_content = re.sub(pattern, "", chunk_content)

                if isinstance(self.chunk_retreiver, Google):
                    context += f"{chunk_content}"

                else:
                    context += f"Source from '{chunk['doc']}':\n------------------------------\n{chunk_content}\n------------------------------\n\n"

            except Exception as e:
                logger.info("Error adding one chunk to context :%s", e)

        self.context = context

    def select_chunk(self, docs: list[Tuple[Document, float]]) -> None:
        """Select chunk according to the score limit"""

        sorted_docs = sorted(docs, key=lambda x: x[1], reverse=True)

        logger.info("Chunk score limit %f", CHUNK_SCORE_LIMIT)

        context_content: list[str] = []

        for doc in sorted_docs:
            try:
                if (
                    doc[1] > CHUNK_SCORE_LIMIT
                    and doc[0].page_content not in context_content
                ):
                    logger.info("ADD CONTEXT with score %f", doc[1])

                    self.context_list.append(
                        ContextChunk(
                            index=len(self.context_list) + 1,
                            content=doc[0].page_content,
                            similarity=doc[1],
                            document=doc[0].metadata["file_name"],
                            url=doc[0].metadata.get("source_url"),
                            sender=doc[0].metadata.get("sender"),
                            cced=doc[0].metadata.get("cced"),
                            bcced=doc[0].metadata.get("bcced"),
                            has_attachment=doc[0].metadata.get("has_attachment"),
                            date_sent=doc[0].metadata.get("date_sent"),
                        )
                    )

                    context_content.append(doc[0].page_content)

                    if doc[0].metadata["file_name"] not in self.doc_used:
                        self.doc_used.append(doc[0].metadata["file_name"])

                if len(self.context_list) >= EnvParam.NB_CHUNK_FOR_CONTEXT:
                    break

            except Exception as e:
                logger.error("Error adding chunk : %s", e)

        logger.warning("Nb chunk for context %d", len(self.context_list))
