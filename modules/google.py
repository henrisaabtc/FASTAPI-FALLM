"""_summary_"""

import json
import asyncio

from typing import Tuple

import httpx

from pydantic import BaseModel

from langchain.docstore.document import Document


from modules import logger


from config.config import EnvParam


class Google(BaseModel):
    """_summary_

    Args:
        BaseModel (_type_): _description_
    """

    async def get_chunks(self, list_queries: list[str]) -> list[Tuple[Document, float]]:
        """_summary_

        Args:
            list_queries (list[str]): _description_

        Returns:
            list[Tuple[Document, float]]: _description_
        """

        try:
            chunks: list[Tuple[Document, float]] = await self.get_web_pages_context(
                list_queries=list_queries
            )

        except Exception as e:
            logger.error("Error getting google context %s", e)

            chunks = []

        return chunks

    async def get_web_pages_context(
        self, list_queries: list[str]
    ) -> list[Tuple[Document, float]]:
        """_summary_

        Args:
            list_queries (list[str]): _description_

        Returns:
            list[Tuple[Document, float]]: _description_
        """

        tasks = [
            self.get_google_context(
                query=query,
            )
            for query in list_queries
        ]

        logger.info("Nb task for google query %d", len(tasks))

        web_pages_contexts: list[list[Tuple[Document, float]]] = await asyncio.gather(
            *tasks
        )

        merged_web_pages_contexts: list[Tuple[Document, float]] = []

        for web_pages_context in web_pages_contexts:
            merged_web_pages_contexts += web_pages_context

        return merged_web_pages_contexts

    async def get_google_context(self, query: str) -> list[Tuple[Document, float]]:
        """Get  google context"""

        doc_serper: list[Tuple[Document, float]] = await self.get_serper(query=query)

        return doc_serper

    async def get_serper(self, query: str) -> list[Tuple[Document, float]]:
        """_summary_

        Args:
            query (str): _description_

        Returns:
            Tuple[list[str], str, str]: _description_
        """

        pages: list[Tuple[Document, float]] = []

        payload = json.dumps({"q": query, "gl": "fr", "hl": "fr"})

        headers = {
            "X-API-KEY": EnvParam.GOOGLE_SERPER_API,
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient() as client:
            responses_search = (
                await client.post(
                    EnvParam.SERPER_URL_SEARCH, headers=headers, data=payload
                )
            ).json()

        keys_to_keep = ["knowledgeGraph", "answerBox", "organic"]

        filtered_response_search = {
            k: responses_search[k] for k in keys_to_keep if k in responses_search
        }

        logger.error(filtered_response_search)

        try:
            answer_box_dict: dict = filtered_response_search["answerBox"]

            list_keys = ["snippet", "snippetHighlighted", "title"]

            answer_box_content: list[str] = []

            for key in list_keys:
                answer_box_content += self.find_keys(result=answer_box_dict, key=key)

            doc_answer_box = Document(
                page_content=query + ": " + " ".join(answer_box_content),
                metadata={"file_name": "link"},
            )

            pages.append((doc_answer_box, 1.0))

        except KeyError as e:
            logger.warning("There is no answer box in Serper response %s", e)

        try:
            knowledge_graph_dict: dict = filtered_response_search["knowledgeGraph"]

            list_keys = ["snippet", "description", "title"]

            knowledge_graph_content: list[str] = []

            for key in list_keys:
                knowledge_graph_content += self.find_keys(
                    result=knowledge_graph_dict, key=key
                )

            page_content = ". ".join(knowledge_graph_content)

            page_content = (
                page_content.replace("\n\n\n", "\n\n")
                .replace("  ", " ")
                .replace("\t\t", "\t")
            )

            doc_knowledge_graph = Document(
                page_content=query + ": " + page_content,
                metadata={"file_name": "link"},
            )

            pages.append((doc_knowledge_graph, 1.0))

        except KeyError as e:
            logger.warning("There is no knowledge in Serper response %s", e)

        try:
            organic_dict: list[dict] = filtered_response_search["organic"]

            for page in organic_dict[:3]:
                try:
                    doc = Document(
                        page_content=f"{query}: {page['title']}. {page['snippet']}".replace(
                            "\n\n", "\n"
                        ),
                        metadata={"file_name": page["link"]},
                    )

                    pages.append((doc, 1.0))

                except KeyError:
                    pass

        except KeyError as e:
            logger.warning("There is no organic in Serper response %s", e)

        async with httpx.AsyncClient() as client:
            responses_place = (
                await client.post(
                    EnvParam.SERPER_URL_PLACES, headers=headers, data=payload
                )
            ).json()["places"]

            responses_place = self.json_place_to_str(dict=responses_place)

            doc_place = Document(
                page_content=f"{query}: {responses_place}",
                metadata={"file_name": "link"},
            )

            pages.append((doc_place, 1.0))

        return pages

    def find_keys(self, result: dict, key) -> list:
        """_summary_

        Args:
            result (dict): _description_
            key (_type_): _description_

        Returns:
            _type_: _description_
        """

        links = []

        for k, v in result.items():
            if k == key:
                links.append(str(v).strip())

            elif isinstance(v, dict):
                links.extend(self.find_keys(v, key))

            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        links.extend(self.find_keys(item, key))

        return links

    def json_place_to_str(self, dict: list[dict]) -> str:
        """_summary_

        Args:
            dict (list[dict]): _description_

        Returns:
            str: _description_
        """

        keys_to_exclude = ["thumbnailUrl", "cid", "placeId", "latitude", "longitude"]

        result = ""

        for item in dict[:2]:
            for key, value in item.items():
                if key not in keys_to_exclude:
                    result += f"{key}: {value}\n"

        result = result.rstrip()

        return result
