"""Module that format text"""


import re
from typing import Tuple

import markdown

from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name, guess_lexer


from pydantic import BaseModel

from modules import logger


class OutputFormater(BaseModel):
    """Class that format code to html"""

    html_folder_path: str

    html_file_name: str

    css_file_name: str

    pattern_start_code: str = r"```[a-zA-Z0-9\+\#\.]+?\n"

    pattern_end_code: str = r"```"

    mismatch_code_block_start: int = 0

    mismatch_code_block_end: int = 0

    offset: int = 0

    def code_formater(self, langage: str, code: str) -> str:
        """Function that format code with librarie lexer"""

        try:
            lexer = get_lexer_by_name(langage)

            return highlight(code, lexer, HtmlFormatter())

        except Exception as e:
            logger.error("Error balising code %s", e)

            return code

    def get_start_code_index(self, text: str) -> list[int]:
        """Function that found start code index in the text"""

        code_matches = re.finditer(self.pattern_start_code, text, re.DOTALL)

        index_start_code = []

        for code_match in code_matches:
            index_start_code.append(code_match.start())

        return index_start_code

    def get_end_code_index(
        self, text: str, list_index_start_code: list[int]
    ) -> list[int]:
        """Function that found end code index in the text"""

        code_matches = re.finditer(self.pattern_end_code, text, re.DOTALL)

        index_end_code = []

        for code_match in code_matches:
            if code_match.start() not in list_index_start_code:
                index_end_code.append(code_match.start() + len(self.pattern_end_code))

        return index_end_code

    def get_code_index(self, text: str) -> Tuple[list[int], list[int]]:
        """Get all the start code index and end code index from text"""

        list_index_start_code: list[int] = self.get_start_code_index(text=text)

        list_index_end_code: list[int] = self.get_end_code_index(
            text=text, list_index_start_code=list_index_start_code
        )

        if len(list_index_start_code) != len(list_index_end_code):
            logger.error(
                "Len start list index %s mismatch len end list index %s",
                str(list_index_start_code),
                str(list_index_end_code),
            )

            logger.info(list_index_start_code)

            logger.info(list_index_end_code)

            if (
                len(list_index_start_code) % 2 == 0
                and len(list_index_end_code) % 2 == 0
            ):
                all_index = sorted(list_index_start_code + list_index_end_code)

                list_index_start_code = []

                list_index_end_code = []

                logger.info(all_index)

                is_fist = True

                for index in all_index:
                    if is_fist:
                        list_index_start_code.append(index)
                    else:
                        list_index_end_code.append(index)

                    is_fist = not is_fist

            else:
                list_index_start_code = []

                list_index_end_code = []

        return list_index_start_code, list_index_end_code

    def get_langage_from_block(self, code_block: str) -> Tuple[str, bool]:
        """Function that found langage type associated to a code part"""

        langage_line = code_block.split("\n")[0]

        langage: str = (
            langage_line.split("```")[-1].strip().replace(" ", "").replace("\n", "")
        )

        code_block_content: str = code_block.replace(("```"), "")

        langage_guessed = guess_lexer(code_block_content).name

        is_langage = True

        if len(langage) == 0:
            langage = langage_guessed

            is_langage = False

        logger.info("Found langage lexer %s", langage_guessed)

        logger.info("Found langage %s", langage)

        return langage, is_langage

    def balise_code(self, langage: str, raw_code_content: str, is_langage: bool) -> str:
        """Convert raw text code to html code"""

        if is_langage:
            code_block_balised: str = f'\n<div class="barre-langage">{langage}</div>\n'

        else:
            code_block_balised: str = f'\n<div class="barre-langage"></div>\n'

        code_block_balised += self.code_formater(langage=langage, code=raw_code_content)

        code_block_balised += "\n<br>\n"

        return code_block_balised

    def mismatch_checker(
        self,
        current_index: int,
        index_start_code: int,
        index_end_code: int,
        list_index_start_code: list[int],
    ) -> bool:
        """Check mismatch in code format"""

        if index_start_code > index_end_code:
            self.mismatch_code_block_start -= 1

            return True

        if current_index < len(list_index_start_code) - 1:
            next_index_start_code = (
                list_index_start_code[
                    current_index + self.mismatch_code_block_start + 1
                ]
                + self.offset
            )

            if index_end_code > next_index_start_code:
                self.mismatch_code_block_end -= 1

                return True

        return False

    def formater_code(self, text: str) -> str:
        """Format code"""

        output_balised_content = text

        offset = 0

        list_index_start_code, list_index_end_code = self.get_code_index(text=text)

        logger.info("List index start code %s", str(list_index_start_code))

        logger.info("List index end code %s", str(list_index_end_code))

        for i, _ in enumerate(list_index_start_code):
            index_start_code: int = (
                list_index_start_code[i + self.mismatch_code_block_start] + offset
            )

            index_end_code = (
                list_index_end_code[i + self.mismatch_code_block_end] + offset
            )

            code_block: str = output_balised_content[index_start_code:index_end_code]

            langage, is_langage = self.get_langage_from_block(code_block=code_block)

            logger.info("Code block %s", code_block)

            if langage not in {"markdown"}:
                raw_code_content: str = "\n".join(code_block.split("\n")[1:-1])

                code_block_balised: str = self.balise_code(
                    langage=langage,
                    raw_code_content=raw_code_content,
                    is_langage=is_langage,
                )

                content_before_code_block: str = output_balised_content[
                    :index_start_code
                ]

                content_after_code_block: str = output_balised_content[index_end_code:]

                output_balised_content = (
                    content_before_code_block
                    + code_block_balised
                    + content_after_code_block
                )

                offset += (len(code_block_balised) + len(content_before_code_block)) - (
                    len(code_block) + len(content_before_code_block)
                )

        return output_balised_content

    def formater_markdown(self, text: str) -> str:
        """Format markdonw"""

        formated_text: str = markdown.markdown(text=text)

        return formated_text

    def formater(self, text: str) -> str:
        """Format code and markdonw"""

        output_balised_begin: str = ""  # f'<!DOCTYPE html>\n<head>\n\t<meta charset="UTF-8"><link rel="stylesheet" type="text/css" href="{self.css_file_name}.css">\n</head>\n<html>\n\t<body>'

        output_balised_end: str = ""  # "\n\t</body>\n</html>"

        output_balised_content: str = self.formater_code(text=text)

        output_balised_content_formated: str = self.formater_markdown(
            text=output_balised_content
        )

        output_balised = (
            output_balised_begin + output_balised_content_formated + output_balised_end
        )

        return output_balised
