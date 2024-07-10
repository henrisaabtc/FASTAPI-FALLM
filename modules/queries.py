"""Create all queries for chunks search"""

import time
import copy
import asyncio
import datetime
from typing import Optional, List, Dict

from pydantic import BaseModel

from langchain.prompts import ChatPromptTemplate

from modules import logger

from modules.llm import (
    create_chat_prompt,
    llm_standalone,
    llm_abstract_query,
    llm_multiquery,
    llm_follow_up_question,
    llm_google_query,
)

from modules.prompt import (
    documents_store_context,
    standalone_instruction,
    multiquery_instruction,
    abstract_instruction,
    follow_up_instructions,
    follow_up_system_prompt,
    google_query_instruction,
    google_query_system_prompt,
)

from modules.embeddings import Embeddings

from modules.extractor import ExtractorQueries


embeddings = Embeddings()


class Queries(BaseModel):
    """Class storing db queries"""

    user_input: str

    memory_list: Optional[List[Dict[str, str]]]

    multi_queries: list[str] = []

    abstract_queries: list[str] = []

    hyde_query: list[str] = []

    all_queries: list[str] = []

    documents_names: str = ""

    is_google: bool = False

    async def get_standalone_user_input(self) -> str:
        """Function creating standalone user input according to the conversation"""

        try:
            instruction: str = f"My query: {self.user_input}\n{standalone_instruction}"

            llm_standalone.system_prompt_str = llm_standalone.system_prompt_str.format(
                documents=self.documents_names
            )

            standalone_prompt: ChatPromptTemplate = create_chat_prompt(
                system_prompt=llm_standalone.system_prompt,
                memory_list=self.memory_list,
                context=documents_store_context.format(documents=self.documents_names),
                instruction=instruction,
            )

            standalone_query: str = await llm_abstract_query.inference(
                prompt=standalone_prompt
            )

            standalone_query.replace("query: ", "").replace("Query: ", "")

            logger.info("Standalone query : %s", standalone_query)

        except Exception as e:
            logger.error("Error creating context queries %s", e)

            standalone_query = self.user_input

        distance: float = await embeddings.embedding_distance(
            text_1=self.user_input, text_2=standalone_query
        )

        logger.info("Distance beetween input and standalone input: %f", distance)

        if distance > 0.1:
            standalone_query = self.user_input

        return standalone_query

    async def get_multi_queries(self, user_input_standalone: str) -> list[str]:
        """Function spliting user query in X sub queries"""

        try:
            instruction: str = (
                f"Query: {user_input_standalone}\n{multiquery_instruction}"
            )

            multi_query_prompt: ChatPromptTemplate = create_chat_prompt(
                system_prompt=llm_multiquery.system_prompt,
                memory_list=self.memory_list,
                context=documents_store_context.format(documents=self.documents_names),
                instruction=instruction,
            )

            multi_queries: list[str] = await llm_multiquery.inference_multi_extractor(
                prompt=multi_query_prompt,
                object=ExtractorQueries,
                object_name="queries",
            )

        except Exception as e:
            logger.error("Error creating multi queries %s", e)

            multi_queries = []

        for i, question in enumerate(multi_queries):
            multi_queries[i] = (
                question.replace("subqueries 1:", "")
                .replace("subqueries 2:", "")
                .replace("subqueries 3:", "")
                .replace("Subqueries 1:", "")
                .replace("Subqueries 2:", "")
                .replace("Subqueries 3:", "")
            )

        logger.info("Multi queries :")

        for query in multi_queries:
            logger.info("\t%s", query)

        return multi_queries

    async def get_abstract_queries(self, user_input_standalone: str) -> list[str]:
        """Function creating abstract query from user query"""

        try:
            instruction: str = f"Query: {user_input_standalone}\n{abstract_instruction}"

            abstract_prompt: ChatPromptTemplate = create_chat_prompt(
                system_prompt=llm_abstract_query.system_prompt,
                memory_list=self.memory_list,
                context=documents_store_context.format(documents=self.documents_names),
                instruction=instruction,
            )

            abstract_queries: list[
                str
            ] = await llm_abstract_query.inference_multi_extractor(
                prompt=abstract_prompt, object=ExtractorQueries, object_name="queries"
            )

            for i, question in enumerate(abstract_queries):
                abstract_queries[i] = (
                    question.replace("stepback query 1:", "")
                    .replace("stepback query 2:", "")
                    .replace("stepback query 3:", "")
                    .replace("Stepback query 1:", "")
                    .replace("Stepback query 2:", "")
                    .replace("Stepback query 3:", "")
                )

            logger.info("Abstract queries :")

            for query in abstract_queries:
                logger.info("\t%s", query)

        except Exception as e:
            logger.error("Error creating abstract queries %s", e)

            abstract_queries = []

        return abstract_queries

    async def get_google_queries(self, user_input_standalone: str) -> list[str]:
        """Function creating abstract query from user query"""

        current_datetime = datetime.datetime.now()
        previous_day = current_datetime - datetime.timedelta(days=1)
        next_day = current_datetime + datetime.timedelta(days=1)

        day_str = current_datetime.strftime("%A, %d %B %Y")
        yesterday_str = previous_day.strftime("%A, %d %B %Y")
        tomorow_str = next_day.strftime("%A, %d %B %Y")

        llm_google_query.system_prompt_str = copy.deepcopy(
            google_query_system_prompt
        ).format(day_str=day_str, tomorow_str=tomorow_str, yesterday_str=yesterday_str)

        instruction = copy.deepcopy(google_query_instruction).format(day_str=day_str)

        logger.warning(llm_google_query.system_prompt_str)

        logger.warning(instruction)

        try:
            google_prompt: ChatPromptTemplate = create_chat_prompt(
                system_prompt=llm_google_query.system_prompt,
                memory_list=None,
                context=user_input_standalone,
                instruction=instruction,
            )

            google_query: str = await llm_google_query.inference(prompt=google_prompt)

            logger.info("Google query : %s", google_query)

        except Exception as e:
            logger.error("Error creating google query %s", e)

            google_query = ""

        return google_query

    async def get_queries(self) -> None:
        """Create queries according to the user input"""

        start_time = time.time()

        if self.is_google:
            multi_queries: list[str] = await self.get_multi_queries(
                user_input_standalone=self.user_input
            )

            task = []

            for query in multi_queries + [self.user_input]:
                task.append(self.get_google_queries(query))

            google_queries: list[str] = await asyncio.gather(*task)

            if all(query == "" for query in google_queries):
                self.all_queries = [self.user_input, self.user_input]

            else:
                self.all_queries = google_queries + [self.user_input]

            logger.info("Google queries: %s", self.all_queries)

        else:
            user_input_standalone: str = await self.get_standalone_user_input()

            multi_queries, abstract_queries = await asyncio.gather(
                self.get_multi_queries(user_input_standalone=user_input_standalone),
                self.get_abstract_queries(user_input_standalone=user_input_standalone),
            )

            self.multi_queries = multi_queries
            self.abstract_queries = abstract_queries
            self.all_queries = (
                [user_input_standalone] + multi_queries + abstract_queries
            )

        logger.info("There is %d queries in list querie", len(self.all_queries))

        if len(self.all_queries) == 0:
            logger.error("There is no queries created")

        logger.warning("Queries:")

        for i, query in enumerate(self.all_queries):
            self.all_queries[i] = query.strip()
            logger.warning("-\t %s", query)

        logger.warning("Query time %f", time.time() - start_time)


async def get_follow_up_questions(
    question: str, answer: str, memory: Optional[List[Dict[str, str]]]
) -> list[str]:
    """get follow up question"""

    follow_up_questions: list[str] = []

    try:
        instruction: str = follow_up_instructions

        llm_follow_up_question.system_prompt_str = copy.deepcopy(
            follow_up_system_prompt
        ).format(question=question, answer=answer)

        follow_up_prompt: ChatPromptTemplate = create_chat_prompt(
            system_prompt=llm_follow_up_question.system_prompt,
            memory_list=memory,
            context=question,
            ai_answer=answer,
            instruction=instruction,
        )

        follow_up_questions: list[
            str
        ] = await llm_follow_up_question.inference_multi_extractor(
            prompt=follow_up_prompt, object=ExtractorQueries, object_name="queries"
        )

    except Exception as e:
        logger.error("Error creating follow up question %s", e)

        follow_up_questions = []

    for i, question in enumerate(follow_up_questions):
        follow_up_questions[i] = (
            question.replace("question 1:", "")
            .replace("question 2:", "")
            .replace("question 3:", "")
        )

    logger.info("Follow up questions :")

    for follow_up_question in follow_up_questions:
        logger.info("\t%s", follow_up_question)

    return follow_up_questions
