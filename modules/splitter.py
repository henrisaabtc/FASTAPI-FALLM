"""Module for documents splitting"""

from typing import Callable

from cleantext import clean

from langchain.docstore.document import Document
from langchain_text_splitters import CharacterTextSplitter, SpacyTextSplitter

from modules import logger

from modules.utils import token_count


class Splitter:
    """Class splitting text in chunk"""

    def __init__(
        self,
        counter: Callable[[str], int],
        chunk_size: int,
        seprators: list[str],
    ) -> None:
        self.counter: Callable[[str], int] = counter

        self.chunk_size: int = chunk_size

        self.separators: list[str] = seprators

    def clean_string(self, text: str, is_clean_text: bool) -> str:
        """Function that remove unwanted char:
        blank lines, two following space, multiple line return"""

        if not is_clean_text:
            return text

        try:
            cleaned_text = clean(
                text,
                fix_unicode=True,
                keep_two_line_breaks=True,
                lower=False,
                to_ascii=True,
                no_emoji=True,
                lang="en",
            )

        except Exception as e:
            logger.error("Error cleaning text before splitting : %s", e)

            cleaned_text = text

        return cleaned_text

    def split(self, docs: list[Document], is_clean_text: bool = True) -> list[Document]:
        """Function which split text"""

        split_docs: list[Document] = []

        for doc in docs:
            logger.info(
                "Splitting %s (%d)",
                doc.metadata["file_name"],
                token_count(doc.page_content),
            )

            chunks: list[str] = [
                self.clean_string(text=doc.page_content, is_clean_text=is_clean_text)
            ]

            logger.info(
                "Raw text clean for %s (%d)",
                doc.metadata["file_name"],
                token_count(chunks[0]),
            )

            logger.info("Nb de n in text doc %d", chunks[0].count("\n"))

            try:
                text_splitter_spacy = SpacyTextSplitter(chunk_size=1000)

                chunks = text_splitter_spacy.split_text(chunks[0])

                logger.info("Text split for %s", doc.metadata["file_name"])

            except Exception as e:
                logger.error("Error creating chunk, will use langchain splitter %s", e)

                text_splitter_tiktoken = CharacterTextSplitter.from_tiktoken_encoder(
                    encoding_name="cl100k_base", chunk_size=300, chunk_overlap=0
                )

                chunks = text_splitter_tiktoken.split_text(chunks[0])

            for chunk in chunks:
                try:
                    logger.info("Chunk size => %d\n", len(chunk))

                    new_doc = Document(page_content=chunk, metadata=doc.metadata)

                    split_docs.append(new_doc)

                except Exception as e:
                    logger.error("Error creating chunk after splitting %s", e)

            logger.info("Chunks stored for %s", doc.metadata["file_name"])

        logger.info("All docs splitted")

        return split_docs
