"""Module adding source to text from context"""

import re
import time
import asyncio
from typing import List, Tuple


from config.config import EnvParam

from modules import logger
from modules.context import Context
from modules.embeddings import Embeddings

if EnvParam.USE_AZURE:
    if EnvParam.EMBEDDING_MODEL_AZURE in {"text-embedding-ada-002"}:
        SOURCE_DISTANCE_LIMIT: float = EnvParam.SOURCE_DISTANCE_LIMIT_EMBEDDINGS_ADA

        SOURCE_DISTANCE_NEIGHBOR: float = (
            EnvParam.SOURCE_DISTANCE_NEIGHBOR_EMBEDDINGS_ADA
        )

    else:
        SOURCE_DISTANCE_LIMIT = EnvParam.SOURCE_DISTANCE_LIMIT_EMBEDDINGS_3

        SOURCE_DISTANCE_NEIGHBOR = EnvParam.SOURCE_DISTANCE_NEIGHBOR_EMBEDDINGS_3

else:
    if EnvParam.EMBEDDINGS_MODEL_OPEN_AI in {"text-embedding-ada-002"}:
        SOURCE_DISTANCE_LIMIT = EnvParam.SOURCE_DISTANCE_LIMIT_EMBEDDINGS_ADA

        SOURCE_DISTANCE_NEIGHBOR = EnvParam.SOURCE_DISTANCE_NEIGHBOR_EMBEDDINGS_ADA

    else:
        SOURCE_DISTANCE_LIMIT = EnvParam.SOURCE_DISTANCE_LIMIT_EMBEDDINGS_3

        SOURCE_DISTANCE_NEIGHBOR = EnvParam.SOURCE_DISTANCE_NEIGHBOR_EMBEDDINGS_3


logger.info("Source distance limit choose : %f", SOURCE_DISTANCE_LIMIT)

logger.info("Source neighbor limit choose : %f", SOURCE_DISTANCE_NEIGHBOR)

embeddings = Embeddings()


class Sourcer:
    """Class that add source to an text according to a context"""

    def __init__(self, context: Context) -> None:
        self.context: Context = context

    def split_and_merge_near_limit(self, text: str, char: str, limit: int) -> list[str]:
        """Split and merge to the limit"""

        # parts = text.split(char)

        # merged_parts = []

        # current_part = parts[0]

        # for part in parts[1:]:
        #     if token_count(current_part + char + part) > limit:
        #         merged_parts.append(current_part)

        #         current_part = part
        #     else:
        #         current_part += char + part

        # merged_parts.append(current_part)

        return [text]  # merged_parts

    async def calculate_similare_source(self, sentence: str) -> List[int]:
        """Function that calculate similare source"""

        score_source_list: List[Tuple[float, int]] = []

        source_match_list: List[int] = []

        for index_source, source in enumerate(self.context.context_list):
            scores_source: list[float] = [10]

            for source_split in self.split_and_merge_near_limit(
                text=source.content, char="\n", limit=500
            ):
                score_source: float = await embeddings.embedding_distance(
                    text_1=sentence, text_2=source_split
                )

                scores_source.append(score_source)

            score_source_list.append((min(scores_source), index_source))

        score_source_list = sorted(score_source_list, key=lambda x: x[0], reverse=False)

        best_source: float = score_source_list[0][0]

        try:
            if best_source < SOURCE_DISTANCE_LIMIT:
                source_match_list.append(score_source_list[0][1])

                for score_following in score_source_list[1:]:
                    if score_following[0] - best_source < SOURCE_DISTANCE_NEIGHBOR:
                        source_match_list.append(score_following[1])

                    if len(source_match_list) >= EnvParam.NB_SOURCE_BY_SENTENCES:
                        break

        except Exception as e:
            logger.info("Error matching source : %s", e)

        return source_match_list

    async def get_similare_source(
        self, sentence: str, index: int
    ) -> Tuple[list[int], int]:
        """Function that get similare source"""

        source_list_match: List[int] = []

        try:
            sentence_clean: str = (
                sentence.strip().replace("\n\n", "\n").replace("  ", " ")
            )

            is_list_index = bool(re.fullmatch(r"[ \n][a-zA-Z0-9]", sentence[-2:]))

            if len(sentence_clean) > 5 and not is_list_index:
                source_list_match = await self.calculate_similare_source(
                    sentence=sentence_clean
                )

        except Exception as e:
            logger.error("Error sourcing sentence %s. %s", sentence, e)

        return source_list_match, index

    def add_source_to_sentence(self, sentence: str, source_list: List[int]) -> str:
        """Add str source to the end of sentence"""

        source_str: str = " ".join([f"[{i+1}]" for i in source_list])

        if len(source_str) > 0:
            sentence = sentence + " " + source_str

        if bool(re.search(r"[a-zA-Z0-9]", sentence)):
            sentence = sentence + ". "

        return sentence

    def split_text(self, text: str) -> List[str]:
        """Split the text in sub sentences"""

        sentences: List[str] = re.split(r"\. ", text)

        sentences_stack: str = ""

        merged_sentences: List[str] = []

        for sentence in sentences:
            if len(sentences_stack) > 0:
                sentences_stack += ". " + sentence
            else:
                sentences_stack += sentence

            if len(sentences_stack) < 200:
                continue

            if (
                sentences_stack[-2] == " "
                or sentences_stack[-3] == " "
                or sentences_stack[-2] == "\n"
                or sentences_stack[-3] == "\n"
                or sentences_stack[-2] == "\t"
                or sentences_stack[-3] == "\t"
            ):
                continue

            merged_sentences.append(sentences_stack)

            sentences_stack = ""

        if len(sentences_stack) > 0:
            merged_sentences.append(sentences_stack)

        logger.info("Nb sentence for sourcing %d", len(merged_sentences))

        return merged_sentences

    async def source(self, text: str) -> str:
        """Function that add source to text"""

        start_source_time = time.time()

        text_sourced: str = ""

        sentences: list[str] = self.split_text(text=text)

        sources_used_list_and_index: list[Tuple[list[int], int]] = []

        try:
            coroutines = [
                self.get_similare_source(sentence=sentence, index=index)
                for index, sentence in enumerate(sentences)
            ]

            sources_used_list_and_index = await asyncio.gather(*coroutines)

            sources_used_list_and_index = sorted(
                sources_used_list_and_index, key=lambda x: x[1]
            )

            sources_used_list: list[list[int]] = [
                source_list[0] for source_list in sources_used_list_and_index
            ]

            logger.info("Source used: %s", str(sources_used_list))

            for source_sentences in sources_used_list:
                for source in source_sentences:
                    self.context.add_context_chunk_to_source(index_chunk=source + 1)

            sourced_sentences = [
                self.add_source_to_sentence(
                    sentence=sentence, source_list=sources_used_list[i]
                )
                for i, sentence in enumerate(sentences)
            ]

            text_sourced = "".join(sourced_sentences)

        except Exception as e:
            logger.error("Error getting source :  %s", e)

            text_sourced = text

        logger.warning("Source time %f", time.time() - start_source_time)

        return text_sourced
