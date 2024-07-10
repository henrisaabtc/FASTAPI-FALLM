"""Module that manage all Llm stuff"""

from __future__ import annotations

from typing import Type, Optional, TypeVar

from pydantic import BaseModel, Field
from langchain.chains.llm import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser
from langchain.schema.messages import (
    HumanMessage,
    AIMessage,
    BaseMessage,
    SystemMessage,
)
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from langchain_community.utils.openai_functions import (
    convert_pydantic_to_openai_function,
)

from langchain_community.callbacks.manager import get_openai_callback

from config.config import EnvParam

from modules import logger, token_counter

from modules.prompt import (
    semantic_review_system_prompt,
    standalone_system_prompt,
    multiquery_system_prompt,
    abstract_system_prompt,
    raw_answer_system_prompt,
    raw_answer_system_prompt_with_context,
    source_system_prompt,
    format_system_prompt,
    final_answer_system_prompt,
    follow_up_system_prompt,
    google_query_system_prompt,
    extract_info_chunk_system_prompt,
)


T_base_model = TypeVar("T_base_model", bound=BaseModel)


class Llm(BaseModel):
    """Class that manage all Llm stuff"""

    is_use_gpt_4: bool

    is_temperature_changeable: bool

    llm_model: str = ""

    llm_temperature: float

    llm_timeout: int

    llm_retries: int

    system_prompt_str: Optional[str] = Field(default=None, alias="system_prompt_str")

    @property
    def llm(self) -> AzureChatOpenAI | ChatOpenAI:
        """ChatOpenAI getter"""

        if self.is_temperature_changeable:
            self.llm_temperature = EnvParam.USER_TEMPERATURE

        if EnvParam.USE_AZURE:
            if EnvParam.USE_GPT_4 and self.is_use_gpt_4:
                self.llm_model = "GPT_4"

                api_key: str = EnvParam.AZURE_OPENAI_API_KEY_GPT_4

                azure_endpoint: str = EnvParam.AZURE_OPENAI_ENDPOINT_GPT_4

                azure_deployment: str = EnvParam.AZURE_GPT_MODEL_DEPLOYMENT_GPT_4

                self.llm_timeout = self.llm_timeout + 50
            else:
                self.llm_model = "GPT_3.5"

                api_key = EnvParam.AZURE_OPENAI_API_KEY

                azure_endpoint = EnvParam.AZURE_OPENAI_ENDPOINT

                azure_deployment = EnvParam.AZURE_GPT_MODEL_DEPLOYMENT

            api_version: str = EnvParam.OPENAI_API_VERSION

            return AzureChatOpenAI(
                azure_deployment=azure_deployment,
                api_key=api_key,
                azure_endpoint=azure_endpoint,
                api_version=api_version,
                model=self.llm_model,
                timeout=self.llm_timeout,
                max_retries=self.llm_retries,
                temperature=self.llm_temperature,
            )

        return ChatOpenAI(
            model=self.llm_model,
            timeout=self.llm_timeout,
            max_retries=self.llm_retries,
            temperature=self.llm_temperature,
        )

    @property
    def system_prompt(self) -> Optional[SystemMessage]:
        """system_prompt getter"""

        if isinstance(self.system_prompt_str, str):
            return SystemMessage(content=self.system_prompt_str)

        elif isinstance(self.system_prompt_str, SystemMessage):
            return self.system_prompt_str

        else:
            return None

    async def inference(
        self, prompt: ChatPromptTemplate, params_prompt: dict = {}
    ) -> str:
        """Function that running Llm"""

        chain = LLMChain(llm=self.llm, prompt=prompt)

        logger.info(
            "Using Llm %s with temperature %f and timeout %d",
            str(self.llm_model),
            self.llm_temperature,
            self.llm_timeout,
        )

        with get_openai_callback() as cb:
            result = (await chain.ainvoke(params_prompt))["text"]

            logger.info("Token used => %d", cb.completion_tokens + cb.prompt_tokens)

            token_counter.token += cb.completion_tokens + cb.prompt_tokens

        return result

    async def inference_multi_extractor(
        self, prompt: ChatPromptTemplate, object: Type[T_base_model], object_name: str
    ) -> list[str]:
        """Function that running Llm and extract output in given format"""

        raw_result: str = await self.inference(prompt=prompt)

        if EnvParam.USE_AZURE:
            raw_result_list = raw_result.split("\n")

            result_list = []

            for result in raw_result_list:
                if len(result) > 4:
                    if (
                        result.split(" ")[0]
                        in {
                            "-",
                            ".",
                            "_",
                            "1.",
                            "2.",
                            "3.",
                            "4.",
                            "5.",
                            "6.",
                            "7.",
                        }
                        and len(result.split(" ")[0]) < 3
                    ):
                        result = " ".join(result.split(" ")[1:])

                    result_list.append(result)

            result_list = result_list[:3]

        else:
            raw_result_prompt: list[BaseMessage] = [HumanMessage(content=raw_result)]

            raw_result_chat_prompt = ChatPromptTemplate.from_messages(raw_result_prompt)

            openai_functions = [convert_pydantic_to_openai_function(object)]

            parser = JsonKeyOutputFunctionsParser(key_name=object_name)

            chain_multi_extractor = (
                raw_result_chat_prompt
                | self.llm.bind(functions=openai_functions)
                | parser
            )

            with get_openai_callback() as cb:
                result_str = await chain_multi_extractor.ainvoke({})

                logger.info("Token used => %d", cb.completion_tokens + cb.prompt_tokens)

                token_counter.token += cb.completion_tokens + cb.prompt_tokens

            result_list: list[str] = [
                str(query["query"]).strip().replace("\n", " ") for query in result_str
            ]

        return result_list


def create_chat_prompt(
    system_prompt: Optional[SystemMessage],
    memory_list: Optional[list[dict[str, str]]],
    context: Optional[str],
    instruction: str,
    ai_answer: Optional[str] = None,
) -> ChatPromptTemplate:
    """Function that create prompt for llm"""

    chat_messages_list: list[BaseMessage] = []

    if system_prompt:
        chat_messages_list.append(system_prompt)

    if memory_list:
        qa_messages_list = [
            message
            for qa in memory_list
            for message in (
                HumanMessage(content=qa["question"]),
                AIMessage(content=qa["answer"]),
            )
        ]

        chat_messages_list += qa_messages_list

    if context:
        chat_messages_list.append(HumanMessage(content=context))

    if ai_answer:
        chat_messages_list.append(AIMessage(content=ai_answer))

    chat_messages_list.append(HumanMessage(content=instruction))

    prompt = ChatPromptTemplate.from_messages(chat_messages_list)

    return prompt


def create_chat_prompt_new(
    system_prompt: Optional[SystemMessage],
    memory_list: Optional[list[dict[str, str]]],
    context: Optional[str],
    instruction: Optional[str],
    message: str,
) -> ChatPromptTemplate:
    """Function that create prompt for llm"""

    chat_messages_list: list[BaseMessage] = []

    if system_prompt:
        chat_messages_list.append(system_prompt)

    if instruction:
        chat_messages_list.append(HumanMessage(content=instruction))

    if memory_list:
        qa_messages_list = [
            message
            for qa in memory_list
            for message in (
                HumanMessage(content=qa["question"]),
                AIMessage(content=qa["answer"]),
            )
        ]

        chat_messages_list += qa_messages_list

    chat_messages_list.append(HumanMessage(content=message))

    if context:
        chat_messages_list.append(HumanMessage(content=context))

    prompt = ChatPromptTemplate.from_messages(chat_messages_list)

    return prompt


llm_semantic_review = Llm(
    is_use_gpt_4=False,
    is_temperature_changeable=False,
    llm_temperature=EnvParam.TEMPERATURE,
    llm_timeout=EnvParam.TIMEOUT,
    llm_retries=EnvParam.NB_RETRY,
    system_prompt_str=semantic_review_system_prompt,
)

llm_standalone = Llm(
    is_use_gpt_4=False,
    is_temperature_changeable=False,
    llm_temperature=EnvParam.TEMPERATURE,
    llm_timeout=EnvParam.TIMEOUT,
    llm_retries=EnvParam.NB_RETRY,
    system_prompt_str=standalone_system_prompt,
)

llm_multiquery = Llm(
    is_use_gpt_4=False,
    is_temperature_changeable=False,
    llm_temperature=EnvParam.TEMPERATURE,
    llm_timeout=EnvParam.TIMEOUT,
    llm_retries=EnvParam.NB_RETRY,
    system_prompt_str=multiquery_system_prompt,
)

llm_abstract_query = Llm(
    is_use_gpt_4=False,
    is_temperature_changeable=False,
    llm_temperature=EnvParam.TEMPERATURE,
    llm_timeout=EnvParam.TIMEOUT,
    llm_retries=EnvParam.NB_RETRY,
    system_prompt_str=abstract_system_prompt,
)

llm_google_query = Llm(
    is_use_gpt_4=False,
    is_temperature_changeable=False,
    llm_temperature=EnvParam.TEMPERATURE,
    llm_timeout=EnvParam.TIMEOUT,
    llm_retries=EnvParam.NB_RETRY,
    system_prompt_str=google_query_system_prompt,
)

llm_extract_info_chunk = Llm(
    is_use_gpt_4=False,
    is_temperature_changeable=False,
    llm_temperature=EnvParam.TEMPERATURE,
    llm_timeout=EnvParam.TIMEOUT,
    llm_retries=EnvParam.NB_RETRY,
    system_prompt_str=extract_info_chunk_system_prompt,
)

llm_raw_answer = Llm(
    is_use_gpt_4=False,
    is_temperature_changeable=True,
    llm_temperature=EnvParam.TEMPERATURE,
    llm_timeout=EnvParam.TIMEOUT,
    llm_retries=EnvParam.NB_RETRY,
    system_prompt_str=raw_answer_system_prompt,
)

llm_raw_answer_with_context = Llm(
    is_use_gpt_4=True,
    is_temperature_changeable=True,
    llm_temperature=EnvParam.TEMPERATURE,
    llm_timeout=EnvParam.TIMEOUT,
    llm_retries=EnvParam.NB_RETRY,
    system_prompt_str=raw_answer_system_prompt_with_context,
)

llm_raw_answer_with_context_review = Llm(
    is_use_gpt_4=True,
    is_temperature_changeable=True,
    llm_temperature=EnvParam.TEMPERATURE,
    llm_timeout=EnvParam.TIMEOUT_REVIEW,
    llm_retries=EnvParam.NB_RETRY,
    system_prompt_str=raw_answer_system_prompt_with_context,
)

llm_source = Llm(
    is_use_gpt_4=False,
    is_temperature_changeable=False,
    llm_temperature=EnvParam.TEMPERATURE,
    llm_timeout=EnvParam.TIMEOUT,
    llm_retries=EnvParam.NB_RETRY,
    system_prompt_str=source_system_prompt,
)

llm_format = Llm(
    is_use_gpt_4=False,
    is_temperature_changeable=False,
    llm_temperature=EnvParam.TEMPERATURE,
    llm_timeout=EnvParam.TIMEOUT,
    llm_retries=EnvParam.NB_RETRY,
    system_prompt_str=format_system_prompt,
)

llm_format_review = Llm(
    is_use_gpt_4=False,
    is_temperature_changeable=False,
    llm_temperature=EnvParam.TEMPERATURE,
    llm_timeout=EnvParam.TIMEOUT_REVIEW,
    llm_retries=EnvParam.NB_RETRY,
    system_prompt_str=format_system_prompt,
)

llm_final_answer = Llm(
    is_use_gpt_4=False,
    is_temperature_changeable=False,
    llm_temperature=EnvParam.TEMPERATURE,
    llm_timeout=EnvParam.TIMEOUT,
    llm_retries=EnvParam.NB_RETRY,
    system_prompt_str=final_answer_system_prompt,
)

llm_follow_up_question = Llm(
    is_use_gpt_4=False,
    is_temperature_changeable=False,
    llm_temperature=EnvParam.TEMPERATURE,
    llm_timeout=EnvParam.TIMEOUT,
    llm_retries=EnvParam.NB_RETRY,
    system_prompt_str=follow_up_system_prompt,
)
