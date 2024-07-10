"""Module that contains Llm output strcture for extraction"""

from typing import List
from pydantic import BaseModel, Field


class SubQuery(BaseModel):
    """One request or one question"""

    query: str = Field(description="One request or one question")


class ExtractorQueries(BaseModel):
    """All the requests and questions"""

    queries: List[SubQuery]
