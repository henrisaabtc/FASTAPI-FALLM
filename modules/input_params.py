"""All the input json params"""

from enum import Enum
from typing import Optional, List, Dict


from pydantic import BaseModel, Field

from typing_extensions import Annotated


class GptModel(Enum):
    """Open AI GPT model"""

    GPT_4_0 = "gpt4.0"

    GPT_3_5_TURBO = "gpt3.5turbo"

    GPT_3_5 = "gpt3.5"


class Deployement(Enum):
    """Possible deployement"""

    CHAT = "chat"

    DOCUMENT = "document"

    SHAREPOINT = "sharepoint"

    WEB = "web"

    EMAIL = "email"

    GLPI = "glpi"


class SearchType(Enum):
    """Possible search type"""

    VECTOR_SIMILARITY_SEARCH = "vector similarity search"

    HYBRID_SEARCH = "hybrid search"

    HYBRID_SEMANTIC_SEARCH = "hybrid semantic search"


class InputParams(BaseModel):
    """Input user params"""

    question: str

    gpt_model: GptModel

    temperature: float

    deployement: Deployement

    userid: str

    useremail: str

    number_of_documents: Annotated[int, Field(strict=True, gt=0, l=6)]

    search_type: SearchType

    chat_history: List[Dict[str, str]]

    documents: Optional[List[Dict[str, str]]] = None


class InputParamsChat(BaseModel):
    """Input user params"""

    question: str

    gpt_model: GptModel

    temperature: float

    deployement: Deployement

    userid: str

    useremail: str

    chat_history: List[Dict[str, str]]

    documents: Optional[List[Dict[str, str]]] = None


class InputParamsWeb(BaseModel):
    """Input user params for web route"""

    question: str

    gpt_model: GptModel

    temperature: float

    deployement: Deployement

    userid: str

    useremail: str

    chat_history: List[Dict[str, str]]

    documents: Optional[List[Dict[str, str]]] = None


class InputParamsEmail(BaseModel):
    """Input user params"""

    question: str

    gpt_model: GptModel

    temperature: float

    deployement: Deployement

    userid: str

    useremail: str

    chat_history: List[Dict[str, str]]

    documents: Optional[List[Dict[str, str]]] = None


class InputParamsGLPI(BaseModel):
    """Input user params"""

    question: str

    gpt_model: GptModel

    temperature: float

    deployement: Deployement

    userid: str

    useremail: str

    chat_history: List[Dict[str, str]]

    documents: Optional[List[Dict[str, str]]] = None
