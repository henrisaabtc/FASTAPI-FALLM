"""Chain of Llm and algo for answer user input"""

import time
import base64
import datetime
from typing import Optional, List, Dict

from langchain.prompts import ChatPromptTemplate

from config.config import EnvParam

from modules import logger
from modules.input_params import (
    InputParams,
    InputParamsChat,
    InputParamsGLPI,
    InputParamsWeb,
    InputParamsEmail,
)
from modules.sourcer import Sourcer
from modules.documents import Documents
from modules.embeddings import Embeddings
from modules.formater import OutputFormater
from modules.context import Context, ContextChunk
from modules.doc_local_search import DocLocalSearch
from modules.azure_ai_vector_search import AzureAIVectorSearch
from modules.google import Google

from modules.utils import window_token_reducer

from modules.llm import (
    create_chat_prompt,
    create_chat_prompt_new,
    llm_format,
    llm_raw_answer,
    llm_raw_answer_with_context,
)

from modules.prompt import (
    raw_answer_context_instruction,
    format_system_prompt,
    format_instruction,
)


if EnvParam.USE_AZURE:
    if EnvParam.EMBEDDING_MODEL_AZURE in {"text-embedding-ada-002"}:
        DISTANCE_LIMIT_HALLUCINATION: float = EnvParam.DISTANCE_LIMIT_HALLUCINATION_ADA

    else:
        DISTANCE_LIMIT_HALLUCINATION = EnvParam.DISTANCE_LIMIT_HALLUCINATION_3

else:
    if EnvParam.EMBEDDINGS_MODEL_OPEN_AI in {"text-embedding-ada-002"}:
        DISTANCE_LIMIT_HALLUCINATION = EnvParam.DISTANCE_LIMIT_HALLUCINATION_ADA

    else:
        DISTANCE_LIMIT_HALLUCINATION = EnvParam.DISTANCE_LIMIT_HALLUCINATION_3

logger.info("Distance limit Hallucination %f", DISTANCE_LIMIT_HALLUCINATION)

if EnvParam.USE_GPT_4:
    LIMIT_TOKEN_CONTEXT = 30000
else:
    LIMIT_TOKEN_CONTEXT = 3500

logger.info("Limit token context %f", LIMIT_TOKEN_CONTEXT)


output_formater = OutputFormater(
    html_folder_path="./html_output",
    html_file_name="code_formated",
    css_file_name="code_formated",
)

embeddings = Embeddings()


class Chain:
    """All the methodes for answer to user input"""

    def __init__(
        self,
        chunk_retreiver: Optional[DocLocalSearch | AzureAIVectorSearch | Google],
        memory_list: Optional[list[dict[str, str]]],
        user_input: str,
        input_params: InputParams | InputParamsChat | InputParamsWeb,
    ) -> None:
        self.memory_list: Optional[list[dict[str, str]]] = memory_list

        self.user_input: str = user_input

        self.context = Context(
            chunk_retreiver=chunk_retreiver,
            memory_list=memory_list,
            user_input=user_input,
        )

        self.input_params = input_params

    def create_documents(self, documents: Documents) -> None:
        """Create documents from base 64"""

        documents.create_documents()

        if len(documents.documents_list) == 0:
            logger.warning("No document created")

    async def get_context(
        self,
    ) -> None:
        """Create the db from the documents and get context according to the conversation"""

        await self.context.get_context()

    def create_json_response(
        self,
        user_input: str,
        answer_html: str,
        answer_raw: str,
        answer_formated: str,
        context_chunks: list[ContextChunk],
        memory_list: Optional[list[dict[str, str]]],
        follow_up_question: list[str],
    ) -> dict:
        """Create the json to sendback from API"""

        if memory_list:
            chat_history: List[Dict[str, str]] = memory_list + [
                {"question": user_input, "answer": answer_raw}
            ]

        else:
            chat_history = [{"question": user_input, "answer": answer_raw}]

        references: list[dict] = []

        for context_chunk in context_chunks:
            if context_chunk.is_used_in_answer:
                if isinstance(self.input_params, InputParamsWeb):
                    source = {
                        "document": context_chunk.document,
                        "content": context_chunk.content,
                        "referenceNumber": context_chunk.index,
                    }

                elif isinstance(self.input_params, InputParams):
                    source = {
                        "document": context_chunk.document,
                        "url": context_chunk.url,
                        "similarity": context_chunk.similarity,
                        "content": context_chunk.content,
                        "referenceNumber": context_chunk.index,
                    }

                elif isinstance(self.input_params, (InputParamsEmail)):
                    source = {
                        "document": context_chunk.document,
                        "similarity": context_chunk.similarity,
                        "content": context_chunk.content,
                        "referenceNumber": context_chunk.index,
                        "sender": context_chunk.sender,
                        "cced": context_chunk.cced,
                        "bcced": context_chunk.bcced,
                        "has_attachment": context_chunk.has_attachment,
                        "date_sent": context_chunk.date_sent,
                    }

                else:
                    source = {
                        "document": context_chunk.document,
                        "similarity": context_chunk.similarity,
                        "content": context_chunk.content,
                        "referenceNumber": context_chunk.index,
                    }

                references.append(source)

        if self.context.queries:
            questions_generated: list[str] = self.context.queries.all_queries

        else:
            questions_generated = []

        logger.warning(answer_html)

        answer_html = base64.b64encode(answer_html.encode("utf-8")).decode("utf-8")

        if isinstance(self.input_params, InputParamsChat):
            response: Dict = {
                "answer_html": answer_html,
                "answer_markdown": answer_formated,
                "answer": answer_raw,
                "gpt_model": self.input_params.gpt_model.value,
                "temperature": self.input_params.temperature,
                "deployement": self.input_params.deployement.value,
                "suggestedQuestions": follow_up_question,
                "chat_history": chat_history,
                "userid": self.input_params.userid,
                "useremail": self.input_params.useremail,
            }

        elif isinstance(self.input_params, InputParams):
            response = {
                "answer_html": answer_html,
                "answer_markdown": answer_formated,
                "answer": answer_raw,
                "questions_generated": questions_generated,
                "gpt_model": self.input_params.gpt_model.value,
                "temperature": self.input_params.temperature,
                "deployement": self.input_params.deployement.value,
                "search_type": self.input_params.search_type.value,
                "suggestedQuestions": follow_up_question,
                "chat_history": chat_history,
                "references": references,
                "userid": self.input_params.userid,
                "useremail": self.input_params.useremail,
            }

        elif isinstance(self.input_params, (InputParamsEmail, InputParamsGLPI)):
            response = {
                "answer_html": answer_html,
                "answer_markdown": answer_formated,
                "answer": answer_raw,
                "questions_generated": questions_generated,
                "gpt_model": self.input_params.gpt_model.value,
                "temperature": self.input_params.temperature,
                "deployement": self.input_params.deployement.value,
                "suggestedQuestions": follow_up_question,
                "chat_history": chat_history,
                "references": references,
                "userid": self.input_params.userid,
                "useremail": self.input_params.useremail,
            }

        elif isinstance(self.input_params, InputParamsWeb):
            response = {
                "answer_html": answer_html,
                "answer_markdown": answer_formated,
                "answer": answer_raw,
                "gpt_model": self.input_params.gpt_model.value,
                "temperature": self.input_params.temperature,
                "deployement": self.input_params.deployement.value,
                "suggestedQuestions": follow_up_question,
                "chat_history": chat_history,
                "references": references,
                "userid": self.input_params.userid,
                "useremail": self.input_params.useremail,
            }

        else:
            response = {}

        return response

    async def get_raw_answer(self) -> str:
        """Function asking Llm for raw answer (no format, no sources)"""

        start_time = time.time()

        if len(self.context.context) > 0:
            self.context.context = window_token_reducer(context=self.context.context)

            context: Optional[str] = f"{self.context.context}"

            if isinstance(self.context.chunk_retreiver, Google):
                current_datetime = datetime.datetime.now()

                day_str = current_datetime.strftime("%A, %d %B %Y")

                message: str = f"Today we are : {day_str}. {self.user_input}"
            else:
                message = self.user_input

            message = message + "\n\nSources:\n" + context

            logger.warning("Message => \n%s", message)

            raw_answer_prompt_with_context: ChatPromptTemplate = create_chat_prompt_new(
                system_prompt=llm_raw_answer_with_context.system_prompt,
                instruction=raw_answer_context_instruction,
                memory_list=self.memory_list,
                message=message,
                context=None,
            )

            try:
                raw_answer_with_context: str = (
                    await llm_raw_answer_with_context.inference(
                        prompt=raw_answer_prompt_with_context
                    )
                )

            except Exception as e:
                logger.error("Error infering Llm raw answer with context, %s", e)

                raw_answer_with_context = EnvParam.ERROR_MESSAGE + " (" + str(e) + ")"

            logger.info("RAW ANSWER: %s", raw_answer_with_context)

            logger.warning("Raw answer time %f", time.time() - start_time)

            return raw_answer_with_context

        instruction: str = f"{self.user_input}"

        raw_answer_prompt: ChatPromptTemplate = create_chat_prompt(
            system_prompt=llm_raw_answer.system_prompt,
            memory_list=self.memory_list,
            context=None,
            instruction=instruction,
        )

        logger.info("RAW ANSWER PROMPT: %s", raw_answer_prompt)

        try:
            raw_answer: str = await llm_raw_answer.inference(prompt=raw_answer_prompt)

        except Exception as e:
            logger.error("Error infering Llm raw answer, %s", e)

            raw_answer = EnvParam.ERROR_MESSAGE + " (" + str(e) + ")"

        logger.info("RAW ANSWER: %s", raw_answer)

        return raw_answer

    async def get_formated_answer(
        self,
        answer: str,
    ) -> str:
        """Function that format the answer in markdown format"""

        start_time = time.time()

        if EnvParam.ERROR_MESSAGE not in answer:
            instruction_format: str = (
                f"Text to format:\n------\n{answer}\n------\n{format_instruction}"
            )

            format_prompt = create_chat_prompt(
                system_prompt=llm_format.system_prompt,
                memory_list=None,
                context=format_system_prompt,
                instruction=instruction_format,
            )

            try:
                structured_answer: str = await llm_format.inference(
                    prompt=format_prompt
                )

                structured_answer = structured_answer.replace("------\n", "")

                structured_answer = structured_answer.replace("\n------", "")

                structured_answer = structured_answer.replace("------", "")

                structured_answer = structured_answer.replace("Text to format:", "")

                structured_answer = structured_answer.replace(format_instruction, "")

                if (
                    len(
                        [
                            line
                            for line in structured_answer.splitlines()
                            if line.strip()
                        ]
                    )
                    == 1
                    and structured_answer.strip()[0] == "#"
                ):
                    structured_answer = structured_answer.strip()[1:]

                if (
                    len(
                        [
                            line
                            for line in structured_answer.splitlines()
                            if line.strip()
                        ]
                    )
                    == 1
                    and structured_answer.strip()[:2] == "##"
                ):
                    structured_answer = structured_answer.strip()[2:]

            except Exception as e:
                logger.error("Error formating answer : %s", e)

                structured_answer = answer

            logger.info("FORMATED ANSWER: %s", structured_answer)

            distance: float = await embeddings.embedding_distance(
                text_1=answer, text_2=structured_answer
            )

            logger.info("Distance beetween answer and format answer: %f", distance)

            if distance > DISTANCE_LIMIT_HALLUCINATION:
                structured_answer = answer

                logger.info("Keep raw answer")

            if "[Formatted text]" in structured_answer:
                structured_answer = answer

            elif "formatted text" in structured_answer:
                structured_answer = answer

            elif "Formatted text" in structured_answer:
                structured_answer = answer

            elif "Heading 1" in structured_answer:
                structured_answer = answer

            elif "heading 1" in structured_answer:
                structured_answer = answer

            elif "[formatted text]" in structured_answer:
                structured_answer = answer

            elif "markdown" in structured_answer or "Markdown" in structured_answer:
                structured_answer = answer

            logger.warning("Format answer time %f", time.time() - start_time)

            return structured_answer

        logger.info("NOT FORMATED ANSWER: %s", answer)

        return answer

    async def get_sourced_answer(self, answer: str) -> str:
        """Function which source an answer with context"""

        if isinstance(self.context.chunk_retreiver, Google):
            self.context.context_list = self.context.context_list[:10]

        sourcer = Sourcer(context=self.context)

        if len(self.context.context) > 0 and EnvParam.ERROR_MESSAGE not in answer:
            answer_sourced = await sourcer.source(text=answer)

        else:
            answer_sourced = answer

        logger.info("SOURCED ANSWER: %s", answer_sourced)

        return answer_sourced

    def markdown_to_html(self, answer: str) -> str:
        """Function which convert a text in markdown format to HTML"""

        answer = answer.replace("   ", "    ")

        answer = answer.replace("\n.\n", "\n")

        answer = answer.replace(" . ", "")

        answer = answer.replace("\n\n\n", "\n\n")

        if answer[-2:] == "\n.":
            answer = answer[:-2]

        if answer[-2:] == " .":
            answer = answer[:-2]

        html_answser: str = output_formater.formater(text=answer)

        html_answser = html_answser.replace("```", "")

        return html_answser
