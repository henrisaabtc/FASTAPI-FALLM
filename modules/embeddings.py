"""Module that manage all emmedings stuff"""

import os
from typing import List, Tuple, Optional

from langchain.evaluation import EmbeddingDistance, EvaluatorType, load_evaluator

from langchain.schema import Document
from langchain.vectorstores.faiss import FAISS
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy


from modules import logger
from modules.splitter import Splitter
from modules.utils import token_count


from config.config import EnvParam


embeddings_handler: AzureOpenAIEmbeddings | OpenAIEmbeddings

if EnvParam.USE_AZURE:
    embeddings_handler = AzureOpenAIEmbeddings(
        azure_deployment=EnvParam.EMBEDDING_DEPLOYMENT,
        api_version=EnvParam.OPENAI_API_VERSION,
        timeout=EnvParam.TIMEOUT_EMBEDDINGS,
    )

else:
    embeddings_handler = OpenAIEmbeddings(model=EnvParam.EMBEDDINGS_MODEL_OPEN_AI)

splitter = Splitter(
    counter=token_count,
    chunk_size=500,
    seprators=[
        "\n\n",
        "\n",
    ],
)


DISTANCE_STRATEGIE = DistanceStrategy.COSINE

IS_NORMALIZE_L2 = False


class Embeddings:
    """Class handling all the embeddings stuff"""

    async def embedding_distance(self, text_1: str, text_2: str) -> float:
        """Function that calcule cosinus distance beetween two text"""

        try:
            evaluator = load_evaluator(
                evaluator=EvaluatorType.EMBEDDING_DISTANCE,
                embeddings=embeddings_handler,
                distance_metric=EmbeddingDistance.COSINE,
            )

            score: float = (
                await evaluator.aevaluate_strings(prediction=text_1, reference=text_2)
            )["score"]

        except Exception as e:
            logger.error("Error calculating distance : %s", e)

            score = 100

        return score

    async def embedding_query(self, query: str) -> List[float]:
        """Convert query to vector"""

        vector: List[float] = embeddings_handler.embed_query(text=query)

        logger.info("Vector for query %s : %s", query, str(vector))

        return vector

    async def create_db(self, chunks: list[Document]) -> Optional[FAISS]:
        """Create FAISS DB with the chunks"""

        try:
            db: FAISS = await FAISS.afrom_documents(
                chunks,
                embeddings_handler,
                distance_strategy=DISTANCE_STRATEGIE,
                normalize_L2=IS_NORMALIZE_L2,
            )

            return db

        except Exception as e:
            logger.error("Error creating FAISS db => %s", e)

            return None

    async def get_chunks_from_db(
        self, query: str, db: FAISS
    ) -> list[Tuple[Document, float]]:
        """Function getting chunk from db according to a query"""

        try:
            chunks: list[
                Tuple[Document, float]
            ] = await db.asimilarity_search_with_relevance_scores(
                query=query, k=EnvParam.NB_CHUNK_BY_QUERY
            )

            sorted_chunks = sorted(chunks, key=lambda x: x[1])

            logger.info("For query : %s, best score is %f", query, sorted_chunks[-1][1])

        except Exception as e:
            logger.error("Error getting chunks for query : %s, %e", query, e)

            chunks = []

        return chunks

    async def select_chunk_with_faiss(
        self, files: list[Document], list_queries: list[str]
    ) -> list[Tuple[Document, float]]:
        """Select chunk from files"""

        chunks: list[Document] = splitter.split(docs=files)

        logger.info("Nb files after spliting %d", len(chunks))

        for chunk in chunks:
            logger.info("Chunk len %d", token_count(chunk.page_content))

        db: Optional[FAISS] = await self.create_db(chunks=chunks)

        if db:
            selected_chunks: list[Tuple[Document, float]] = []

            for query in list_queries:
                selected_chunks += await self.get_chunks_from_db(query=query, db=db)

            selected_chunks_sorted: list[Tuple[Document, float]] = sorted(
                selected_chunks, key=lambda x: x[1]
            )

            return selected_chunks_sorted

        return []
