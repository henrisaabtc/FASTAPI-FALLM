from typing import Tuple

from pydantic import BaseModel

from langchain.docstore.document import Document

from modules.documents import Documents

from modules.embeddings import Embeddings

embeddings = Embeddings()


class DocLocalSearch(BaseModel):
    """Handle all azure search query"""

    documents: Documents

    async def get_chunks(self, list_queries: list[str]) -> list[Tuple[Document, float]]:
        """Function that get chunk and create string context for semantic"""

        chunks: list[Tuple[Document, float]] = await embeddings.select_chunk_with_faiss(
            files=self.documents.documents_list, list_queries=list_queries
        )

        return chunks
