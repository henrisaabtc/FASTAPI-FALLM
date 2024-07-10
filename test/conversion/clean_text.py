from cleantext import clean

text = """This   is a sAmple   text with WHITESPACE, ¿•: umbers 1234, and 😊 emojis!!!



Fisrt tilte

How is it"""
cleaned_text = clean(
    text,
    fix_unicode=True,
    keep_two_line_breaks=True,
    to_ascii=True,
    no_emoji=True,
    lang="en",
)
print(cleaned_text)
