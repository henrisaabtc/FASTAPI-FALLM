"""Modules which contains utils functions"""

import tiktoken

from bs4 import BeautifulSoup, NavigableString

from modules import logger

from config.config import EnvParam


def token_count(text: str) -> int:
    """Function that count token in text"""

    encoding = tiktoken.get_encoding("cl100k_base")

    nb_token: int = len(encoding.encode(text))

    return nb_token


def reduce_until_token(text: str, max_token: int) -> str:
    """Cut text to match max_token size"""

    nb_token = token_count(text=text)

    while nb_token > int(max_token - 10):
        pourcentage = 0.80

        new_limit = int(len(text) * pourcentage)

        text = text[:new_limit]

        nb_token = token_count(text=text)

    return text


def window_token_reducer(context: str) -> str:
    """Check context size and reduce according to llm windiw size"""

    if token_count(context) > EnvParam.MAX_TOKEN_CONTEXT:
        old_len: int = token_count(context)

        context = reduce_until_token(
            text=context,
            max_token=EnvParam.MAX_TOKEN_CONTEXT,
        )

        logger.warning(
            "Reduce context from %d to %d",
            old_len,
            token_count(context),
        )

    return context


def save_result(index_question: int, html_answer: str, user_input: str, context: str):
    """Function that saves the QA result to a local folder."""

    soup = BeautifulSoup(html_answer, "html.parser")

    user_question_html = soup.new_tag("div")
    user_question_h1 = soup.new_tag("h1")
    user_question_h1.append(NavigableString("User question:"))
    user_question_html.append(user_question_h1)

    user_question_p = soup.new_tag("p")
    user_question_p.append(NavigableString(user_input))
    user_question_html.append(user_question_p)

    ai_answer_html = soup.new_tag("div")
    ai_answer_h1 = soup.new_tag("h1")
    ai_answer_h1.append(NavigableString("AI answer:"))
    ai_answer_html.append(ai_answer_h1)

    for content in soup.body.contents:
        ai_answer_html.append(content)

    source_html = soup.new_tag("div")
    source_h1 = soup.new_tag("h1")
    source_h1.append(NavigableString("Sources:"))
    source_html.append(source_h1)

    source_p = soup.new_tag("p")
    source_p.append(NavigableString(context))
    source_html.append(source_p)

    soup.body.clear()
    soup.body.append(user_question_html)
    soup.body.append(ai_answer_html)
    soup.body.append(source_html)

    final_html = str(soup)
    result_file = f"./test/results/{index_question}.html"
    with open(result_file, "w", encoding="utf-8") as f:
        f.write(final_html)


if __name__ == "__main__":
    save_result(
        1,
        "<html><body><p>Original body content.</p></body></html>",
        "How does BeautifulSoup work?",
        "Contextual information here.",
    )
