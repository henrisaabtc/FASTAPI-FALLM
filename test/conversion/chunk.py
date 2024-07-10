import fitz
from cleantext import clean
from pymupdf_rag import to_markdown

from langchain_text_splitters import SpacyTextSplitter

text_splitter = SpacyTextSplitter(chunk_size=1000)


path = "test/conversion/BASE_CONNAISSANCES_ARBS.pdf"

doc = fitz.open(path)

pdf_txt = to_markdown(doc)

cleaned_text = clean(
    pdf_txt,
    fix_unicode=True,
    keep_two_line_breaks=True,
    to_ascii=True,
    no_emoji=True,
    lang="en",
)

texts = text_splitter.split_text(pdf_txt)

for chunk in texts:
    print("\n\n\n///////////////////////////////////////\n\n")
    print(chunk)
