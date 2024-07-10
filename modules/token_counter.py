"""Count all the token used"""

from pydantic import BaseModel


class TokenCounter(BaseModel):
    """_summary_

    Args:
        BaseModel (_type_): _description_
    """

    token: int
