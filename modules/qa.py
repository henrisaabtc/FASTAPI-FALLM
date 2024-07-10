"""Module handling all the QA chain"""

import os
import time
from typing import Optional, List, Dict

from datetime import datetime

from pydantic import ValidationError

from azure.monitor.events.extension import track_event

from config.config import EnvParam

from modules import logger, token_counter

from modules.chain import Chain
from modules.documents import Documents
from modules.exception import NoDocumentsException
from modules.doc_local_search import DocLocalSearch
from modules.azure_ai_vector_search import AzureAIVectorSearch
from modules.azure_ai_vector_search_email import AzureAIVectorSearchEmail
from modules.azure_ai_vector_search_glpi import AzureAIVectorSearchGLPI
from modules.google import Google
from modules.input_params import (
    InputParams,
    Deployement,
    GptModel,
    InputParamsChat,
    InputParamsWeb,
    InputParamsEmail,
    InputParamsGLPI,
)
from modules.queries import get_follow_up_questions


class QA:
    """Class that answer user message"""

    def __init__(
        self,
        input_params: (
            InputParams
            | InputParamsChat
            | InputParamsWeb
            | InputParamsEmail
            | InputParamsGLPI
        ),
    ) -> None:
        token_counter.token = 0

        self.input_params: (
            InputParams | InputParamsChat | InputParamsWeb | InputParamsGLPI
        ) = input_params

        self.user_input: str = input_params.question

        self.memory_list: Optional[List[Dict[str, str]]] = input_params.chat_history

        self.documents = Documents(base64_documents=input_params.documents)

        chunk_retreiver: Optional[
            DocLocalSearch
            | AzureAIVectorSearch
            | Google
            | AzureAIVectorSearchEmail
            | AzureAIVectorSearchGLPI
        ]

        if self.input_params.deployement == Deployement.SHAREPOINT:
            chunk_retreiver = AzureAIVectorSearch()

        elif self.input_params.deployement == Deployement.EMAIL:
            chunk_retreiver = AzureAIVectorSearchEmail()

        elif self.input_params.deployement == Deployement.GLPI:
            chunk_retreiver = AzureAIVectorSearchGLPI()

        elif self.input_params.deployement == Deployement.DOCUMENT:
            if not self.input_params.documents:
                raise NoDocumentsException()

            chunk_retreiver = DocLocalSearch(documents=self.documents)

        elif self.input_params.deployement == Deployement.CHAT:
            chunk_retreiver = None

        elif self.input_params.deployement == Deployement.WEB:
            chunk_retreiver = Google()

        else:
            raise ValidationError

        self.chain = Chain(
            input_params=self.input_params,
            chunk_retreiver=chunk_retreiver,
            user_input=self.user_input,
            memory_list=self.memory_list,
        )

    def input_param_to_env(self):
        """Setup param from json to env"""

        if self.input_params.gpt_model == GptModel.GPT_4_0:
            EnvParam.USE_GPT_4 = True

        EnvParam.USER_TEMPERATURE = self.input_params.temperature

        if self.input_params.deployement in {
            Deployement.DOCUMENT,
            Deployement.SHAREPOINT,
        }:
            EnvParam.NB_FILES_FOR_CHUNKS = self.input_params.number_of_documents

        if self.input_params.deployement in {Deployement.WEB}:
            EnvParam.NB_CHUNK_FOR_CONTEXT = 100

    async def run(self) -> Dict:
        """Function that create raw answser from user input and context, then add format and sources"""

        start_time = time.time()

        self.input_param_to_env()

        if self.input_params.deployement == Deployement.DOCUMENT:
            self.chain.create_documents(documents=self.documents)

        if self.input_params.deployement in {
            Deployement.DOCUMENT,
            Deployement.SHAREPOINT,
            Deployement.WEB,
            Deployement.EMAIL,
            Deployement.GLPI,
        }:
            await self.chain.get_context()

        answer_raw: str = await self.chain.get_raw_answer()

        answer_formated: str = await self.chain.get_formated_answer(
            answer=answer_raw,
        )

        # answer_formated: str = answer_raw

        if self.input_params.deployement in {
            Deployement.DOCUMENT,
            Deployement.SHAREPOINT,
            Deployement.EMAIL,
            Deployement.WEB,
            Deployement.GLPI,
        }:
            answer_sourced = await self.chain.get_sourced_answer(answer=answer_formated)

        else:
            answer_sourced = answer_formated

        answer_html: str = self.chain.markdown_to_html(answer=answer_sourced)

        follow_up_question: list[str] = await get_follow_up_questions(
            question=self.user_input, answer=answer_raw, memory=self.memory_list
        )

        response: Dict = self.chain.create_json_response(
            user_input=self.user_input,
            answer_html=answer_html,
            answer_raw=answer_raw,
            answer_formated=answer_sourced,
            context_chunks=self.chain.context.context_list,
            memory_list=self.memory_list,
            follow_up_question=follow_up_question,
        )

        now = datetime.now()

        timestamp = (
            datetime.now().strftime("%m/%d/%Y, %I:%M:%S")
            + "."
            + str(now.microsecond // 1000).zfill(3)
            + " "
            + now.strftime("%p")
        )

        # track_event(
        #    "Response Details",
        #    {
        #        "gpt_model": response["gpt_model"],
        #        "temperature": response["temperature"],
        #        "deployment": response["deployement"],
        #        "userid": response["userid"],
        #        "useremail": response["useremail"],
        #        "tokensUsed": str(token_counter.token),
        #        "timestamp": timestamp,
        #    },
        # )

        logger.warning("Total time %f", time.time() - start_time)

        return response
