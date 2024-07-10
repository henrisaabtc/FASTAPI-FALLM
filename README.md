## Need :

- Visual Studio Code
- Azure function extension for Vs Code : https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-azurefunctions
- Connect to Azure via Vs code with henri.saab@alt-up.com account

## LOCAL

⁠Clone the repo to a local folder.
⁠Open the folder in Vs Code.
⁠Create a virtual environment in the folder with the command : python3.10 -m venv .venv (You must use a python version higher than 3.10)
⁠Copy the .env file in ./config/.env (see below)
Debug the function localy => click on execute (top bar) -> click on start debugging. This will first install the python packages in the virtual environment (.venv).
⁠You can also manage the virtual environment with the poetry executable.
⁠To add a python package to the env, add its name manually to the requirement.txt file.
⁠After launch, 4 routes are deployed on localhost port 7071 under route /api. Examples are shared in the file "Friday-gpt.postman_collection.json".
⁠The python code entry points can be found in "function_app.py".
⁠Configuration parameters can be found in ./config/app.ini or ./config/.env

Optionnal :
In order to configure azure function go get telemetary data
please add the instrumentation key to the localsetting.json
"APPLICATIONINSIGHTS_CONNECTION_STRING" :"InstrumentationKey=to_replace;IngestionEndpoint=to_replace;LiveEndpoint=to_replace;ApplicationId=to_replace »

## DEPLOY

- In the Azure extension tab, select :
  -> Resources
  -> Azure subscription 1
  -> Function App
  -> FA-LLM => Right-click => Deploy To Function App
- After deploy succeed, 4 routes are deployed on https://fa-llm.azurewebsites.net/api. Examples are shared in the file "Friday-gpt.postman_collection.json".

## .env

AZURE_OPENAI_ENDPOINT=

AZURE_OPENAI_ENDPOINT_GPT_4=

AZURE_OPENAI_API_KEY=

AZURE_OPENAI_API_KEY_GPT_4=

AZURE_GPT_MODEL_DEPLOYMENT=

AZURE_GPT_MODEL_DEPLOYMENT_GPT_4=

EMBEDDING_MODEL_AZURE=

EMBEDDING_DEPLOYMENT=

OPENAI_API_VERSION=

OPENAI_API_KEY=

AISEARCH_ENDPOINT=

AISEARCH_KEY=

AISEARCH_INDEXNAME=

SEMANTIC_CONFIG_NAME=

AISEARCH_INDEXNAME_VECTOR=

SEMANTIC_CONFIG_NAME_VECTOR=

USE_AZURE=

USE_GPT_4=False

GOOGLE_SERPER_API=

APPLICATIONINSIGHTS_CONNECTION_STRING=
