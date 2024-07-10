# Ensure this is the first import
import os
import json
import traceback
import certifi
import logging
from fastapi import FastAPI, Request, HTTPException
from pydantic import ValidationError
from modules import logger
from modules.qa import QA
from modules.input_params import (
    InputParams,
    InputParamsChat,
    InputParamsWeb,
    InputParamsEmail,
    InputParamsGLPI,
)
from dotenv import load_dotenv

load_dotenv()

# Ensure requests use certifi certificates
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# Add debugging statement to print the loaded environment variable
print("AZURE_OPENAI_API_KEY:", os.getenv("AZURE_OPENAI_API_KEY"))
app = FastAPI()

# Set up basic logging
logging.basicConfig(level=logging.INFO)


@app.post("/chat")
async def chat(request: Request):
    """Chat endpoint"""
    logging.info("Chat route triggered")
    try:
        req_body = await request.json()
        input_params = InputParamsChat.model_validate(req_body)
        qa = QA(input_params=input_params)
        response = await qa.run()
        return response
    except ValidationError as e:
        logging.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=e.errors())
    except Exception as e:
        error_trace = traceback.format_exc()
        logging.error(f"Unhandled exception: {error_trace}")
        raise HTTPException(status_code=500, detail=str(error_trace))


@app.post("/document")
async def document(request: Request):
    """Document insight endpoint"""
    logging.info("Document insight route triggered")
    try:
        req_body = await request.json()
        input_params = InputParams.model_validate(req_body)
        qa = QA(input_params=input_params)
        response = await qa.run()
        return response
    except ValidationError as e:
        logging.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=e.errors())
    except Exception as e:
        error_trace = traceback.format_exc()
        logging.error(f"Unhandled exception: {error_trace}")
        raise HTTPException(status_code=500, detail=str(error_trace))


@app.post("/sharepoint")
async def sharepoint(request: Request):
    """Sharepoint insight endpoint"""
    logging.info("Sharepoint insight route triggered")
    try:
        req_body = await request.json()
        input_params = InputParams.model_validate(req_body)
        qa = QA(input_params=input_params)
        response = await qa.run()
        return response
    except ValidationError as e:
        logging.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=e.errors())
    except Exception as e:
        error_trace = traceback.format_exc()
        logging.error(f"Unhandled exception: {error_trace}")
        raise HTTPException(status_code=500, detail=str(error_trace))


@app.post("/web")
async def web(request: Request):
    """Web insight endpoint"""
    logging.warning("Web route triggered")
    try:
        req_body = await request.json()
        input_params = InputParamsWeb.model_validate(req_body)
        qa = QA(input_params=input_params)
        response = await qa.run()
        return response
    except ValidationError as e:
        logging.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=e.errors())
    except Exception as e:
        error_trace = traceback.format_exc()
        logging.error(f"Unhandled exception: {error_trace}")
        raise HTTPException(status_code=500, detail=str(error_trace))


@app.post("/email")
async def email(request: Request):
    """Email insight endpoint"""
    logging.info("Email insight route triggered")
    try:
        req_body = await request.json()
        input_params = InputParamsEmail.model_validate(req_body)
        qa = QA(input_params=input_params)
        response = await qa.run()
        return response
    except ValidationError as e:
        logging.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=e.errors())
    except Exception as e:
        error_trace = traceback.format_exc()
        logging.error(f"Unhandled exception: {error_trace}")
        raise HTTPException(status_code=500, detail=str(error_trace))


@app.post("/glpi")
async def glpi(request: Request):
    """GLPI insight endpoint"""
    logging.info("GLPI insight route triggered")
    try:
        req_body = await request.json()
        input_params = InputParamsGLPI.model_validate(req_body)
        qa = QA(input_params=input_params)
        response = await qa.run()
        return response
    except ValidationError as e:
        logging.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=e.errors())
    except Exception as e:
        error_trace = traceback.format_exc()
        logging.error(f"Unhandled exception: {error_trace}")
        raise HTTPException(status_code=500, detail=str(error_trace))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
