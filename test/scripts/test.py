import json
import time
from numpy import mean
import requests

import pandas as pd
from bs4 import BeautifulSoup, NavigableString

TEST_FILE = "test/files/test_questions.xlsx"

API_URL_LOCAL = "http://localhost:7071/api/vector_insight"

API_URL_DEPLOYED = "https://fa-fridaygpt-docinsight.azurewebsites.net/api/doc_insight"

df = pd.read_excel(TEST_FILE)

questions_list = df.iloc[:, 0].astype(str).tolist()

list_time_exec: list[float] = []

for i, question in enumerate(questions_list):
    print(f"\n\nQuestion ({i}/{len(questions_list)}): {question}")

    payload = json.dumps(
        {
            "question": question,
            "USE_GPT_4": "False",
            "TEMPERATURE": 0,
            "NB_FILES_FOR_CHUNKS": 2,
            "chat_history": [
                {
                    "question": "Do you ever met Paul and Anis ?",
                    "answer": "I never met them",
                },
                {
                    "question": "Which one is a cardiologist?",
                    "answer": "Dr. Paul is the cardiologist mentioned in the sources provided",
                },
            ],
        }
    )
    headers = {"Content-Type": "application/json"}

    start_time = time.time()

    response = requests.request("GET", API_URL_LOCAL, headers=headers, data=payload)

    total_time = time.time() - start_time

    print("Total time %f", total_time)

    list_time_exec.append(total_time)

    json_output = response.json()

    user_input = question

    html_answer = json_output["answer"]

    references: dict = json_output["references"]

    references_list: list = []

    for reference in references:
        references_list.append(str(reference))

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

    for reference in references_list:
        source_p = soup.new_tag("p")
        source_p.append(NavigableString(str(reference)))
        source_html.append(source_p)

    soup.body.clear()
    soup.body.append(user_question_html)
    soup.body.append(ai_answer_html)
    soup.body.append(source_html)

    final_html = str(soup)
    result_file = f"./test/files/results_html/{i}.html"
    with open(result_file, "w", encoding="utf-8") as f:
        f.write(final_html)

print(list_time_exec)
print(min(list_time_exec))
print(max(list_time_exec))
print(mean(list_time_exec))
