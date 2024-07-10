import requests
import json

url = "https://localhost:7071/api/email"

payload = json.dumps(
    {
        "question": "Comment vérifier la cohérence des statistiques de réception des lignes contrôlées au PDA par rapport aux bons intégrés en stock?",
        "gpt_model": "gpt3.5turbo",
        "temperature": 0,
        "deployement": "email",
        "userid": "012345",
        "useremail": "clement.remillieux@gmail.com",
        "chat_history": [],
    }
)
headers = {"Content-Type": "application/json"}

response = requests.request("GET", url, headers=headers, data=payload, verify=False)

print(response.text)
