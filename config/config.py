"""Module with all config/env stuff"""

import os

import configparser
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


config = configparser.ConfigParser()

config.read("config/app.ini")


def load_param_str_config(section: str, param_name: str) -> str:
    """Load param from .ini file"""

    param: str = config.get(section, param_name)

    return param


def load_params_env_file(name: str) -> str:
    """Load param from .env file"""

    return str(os.getenv(name))


class EnvParam:
    """Class that contain all the loaded param/env var"""

    LOGGER_NAME: Optional[str] = str(
        load_param_str_config(section="app", param_name="LOGGER_NAME")
    )

    if LOGGER_NAME == "None":
        LOGGER_NAME = None

    SOURCE_DISTANCE_LIMIT_EMBEDDINGS_3: float = float(
        load_param_str_config(
            section="source", param_name="SOURCE_DISTANCE_LIMIT_EMBEDDINGS_3"
        )
    )

    SOURCE_DISTANCE_NEIGHBOR_EMBEDDINGS_3: float = float(
        load_param_str_config(
            section="source", param_name="SOURCE_DISTANCE_NEIGHBOR_EMBEDDINGS_3"
        )
    )

    SOURCE_DISTANCE_LIMIT_EMBEDDINGS_ADA: float = float(
        load_param_str_config(
            section="source", param_name="SOURCE_DISTANCE_LIMIT_EMBEDDINGS_ADA"
        )
    )

    SOURCE_DISTANCE_NEIGHBOR_EMBEDDINGS_ADA: float = float(
        load_param_str_config(
            section="source", param_name="SOURCE_DISTANCE_NEIGHBOR_EMBEDDINGS_ADA"
        )
    )

    NB_SOURCE_BY_SENTENCES: int = int(
        load_param_str_config(section="source", param_name="NB_SOURCE_BY_SENTENCES")
    )

    GPT_3_5_MODEL: str = str(
        load_param_str_config(section="llm", param_name="GPT_3_5_MODEL")
    )

    ERROR_MESSAGE: str = str(
        load_param_str_config(section="llm", param_name="ERROR_MESSAGE")
    )

    TIMEOUT: int = int(load_param_str_config(section="llm", param_name="TIMEOUT"))

    TIMEOUT_REVIEW: int = int(
        load_param_str_config(section="llm", param_name="TIMEOUT_REVIEW")
    )

    NB_RETRY: int = int(load_param_str_config(section="llm", param_name="NB_RETRY"))

    TEMPERATURE: float = float(
        load_param_str_config(section="llm", param_name="TEMPERATURE")
    )

    USER_TEMPERATURE: float = float(
        load_param_str_config(section="llm", param_name="USER_TEMPERATURE")
    )

    DISTANCE_LIMIT_HALLUCINATION_ADA: float = float(
        load_param_str_config(
            section="llm", param_name="DISTANCE_LIMIT_HALLUCINATION_ADA"
        )
    )

    DISTANCE_LIMIT_HALLUCINATION_3: float = float(
        load_param_str_config(
            section="llm", param_name="DISTANCE_LIMIT_HALLUCINATION_3"
        )
    )

    MAX_TOKEN_CONTEXT: int = int(
        load_param_str_config(section="llm", param_name="MAX_TOKEN_CONTEXT")
    )

    EMBEDDINGS_MODEL_OPEN_AI: str = str(
        load_param_str_config(
            section="embeddings", param_name="EMBEDDINGS_MODEL_OPEN_AI"
        )
    )

    TIMEOUT_EMBEDDINGS: str = str(
        load_param_str_config(section="embeddings", param_name="TIMEOUT_EMBEDDINGS")
    )

    NB_CHUNK_FOR_CONTEXT: int = int(
        load_param_str_config(section="embeddings", param_name="NB_CHUNK_FOR_CONTEXT")
    )

    NB_CHUNK_BY_QUERY: int = int(
        load_param_str_config(section="embeddings", param_name="NB_CHUNK_BY_QUERY")
    )

    NB_FILE_BY_QUERY: int = int(
        load_param_str_config(section="embeddings", param_name="NB_FILE_BY_QUERY")
    )

    NB_FILES_FOR_CHUNKS: int = int(
        load_param_str_config(section="embeddings", param_name="NB_FILES_FOR_CHUNKS")
    )

    CHUNK_SCORE_LIMIT_EMBEDDINGS_3: float = float(
        load_param_str_config(
            section="embeddings", param_name="CHUNK_SCORE_LIMIT_EMBEDDINGS_3"
        )
    )

    CHUNK_SCORE_LIMIT_EMBEDDINGS_ADA: float = float(
        load_param_str_config(
            section="embeddings", param_name="CHUNK_SCORE_LIMIT_EMBEDDINGS_ADA"
        )
    )

    FAISS_DB_PATH: str = str(
        load_param_str_config(section="embeddings", param_name="FAISS_DB_PATH")
    )

    DOC_RAG_PATH: str = str(
        load_param_str_config(section="embeddings", param_name="DOC_RAG_PATH")
    )

    INDEX_NAME: str = str(
        load_param_str_config(section="embeddings", param_name="INDEX_NAME")
    )

    CHUNK_SIZE: int = int(
        load_param_str_config(section="embeddings", param_name="CHUNK_SIZE")
    )

    CONTENT_VECTOR: str = str(
        load_param_str_config(section="azure_search", param_name="CONTENT_VECTOR")
    )

    CONTENT: str = str(
        load_param_str_config(section="azure_search", param_name="CONTENT")
    )

    METADTA: str = str(
        load_param_str_config(section="azure_search", param_name="METADTA")
    )

    USER_INPUT_PARAM_NAME: str = str(
        load_param_str_config(
            section="json_payload", param_name="USER_INPUT_PARAM_NAME"
        )
    )

    CHAT_HISTORY_PARAM_NAME: str = str(
        load_param_str_config(
            section="json_payload", param_name="CHAT_HISTORY_PARAM_NAME"
        )
    )

    DOCUMENTS_PARAM_NAME: str = str(
        load_param_str_config(section="json_payload", param_name="DOCUMENTS_PARAM_NAME")
    )

    DOCUMENT_METADATA_NAME: str = str(
        load_param_str_config(
            section="json_payload", param_name="DOCUMENT_METADATA_NAME"
        )
    )

    BASE64_PARAM_NAME: str = str(
        load_param_str_config(section="json_payload", param_name="BASE64_PARAM_NAME")
    )

    USE_GPT_4_PARAM_NAME: str = str(
        load_param_str_config(section="json_payload", param_name="USE_GPT_4_PARAM_NAME")
    )

    TEMPERATURE_PARAM_NAME: str = str(
        load_param_str_config(
            section="json_payload", param_name="TEMPERATURE_PARAM_NAME"
        )
    )

    NB_FILES_FOR_CHUNKS_PARAM_NAME: str = str(
        load_param_str_config(
            section="json_payload", param_name="NB_FILES_FOR_CHUNKS_PARAM_NAME"
        )
    )

    NB_URL_BY_QUERY: int = int(
        load_param_str_config(section="google", param_name="NB_URL_BY_QUERY")
    )

    SERPER_URL_SEARCH: str = str(
        load_param_str_config(section="google", param_name="SERPER_URL_SEARCH")
    )

    SERPER_URL_PLACES: str = str(
        load_param_str_config(section="google", param_name="SERPER_URL_PLACES")
    )

    AZURE_OPENAI_ENDPOINT: str = str(load_params_env_file(name="AZURE_OPENAI_ENDPOINT"))

    AZURE_OPENAI_API_KEY: str = str(load_params_env_file(name="AZURE_OPENAI_API_KEY"))

    AZURE_OPENAI_ENDPOINT_GPT_4: str = str(
        load_params_env_file(name="AZURE_OPENAI_ENDPOINT_GPT_4")
    )

    AZURE_OPENAI_API_KEY_GPT_4: str = str(
        load_params_env_file(name="AZURE_OPENAI_API_KEY_GPT_4")
    )

    AZURE_GPT_MODEL_DEPLOYMENT: str = str(
        load_params_env_file(name="AZURE_GPT_MODEL_DEPLOYMENT")
    )

    AZURE_GPT_MODEL_DEPLOYMENT_GPT_4: str = str(
        load_params_env_file(name="AZURE_GPT_MODEL_DEPLOYMENT_GPT_4")
    )

    EMBEDDING_MODEL_AZURE: str = str(load_params_env_file(name="EMBEDDING_MODEL_AZURE"))

    EMBEDDING_DEPLOYMENT: str = str(load_params_env_file(name="EMBEDDING_DEPLOYMENT"))

    OPENAI_API_VERSION: str = str(load_params_env_file(name="OPENAI_API_VERSION"))

    AZURE_AI_SEARCH_ENDPOINT: str = str(
        load_params_env_file(name="AZURE_AI_SEARCH_ENDPOINT")
    )

    AZURE_AI_SEARCH_KEY: str = str(load_params_env_file(name="AZURE_AI_SEARCH_KEY"))

    AZURE_AI_SEARCH_INDEX_NAME_SHAREPOINT: str = str(
        load_params_env_file(name="AZURE_AI_SEARCH_INDEX_NAME_SHAREPOINT")
    )

    AZURE_AI_SEARCH_INDEX_NAME_GLPI: str = str(
        load_params_env_file(name="AZURE_AI_SEARCH_INDEX_NAME_GLPI")
    )

    AZURE_AI_SEARCH_INDEX_NAME_MAIL: str = str(
        load_params_env_file(name="AZURE_AI_SEARCH_INDEX_NAME_MAIL")
    )

    SEMANTIC_CONFIG_NAME_VECTOR: str = str(
        load_params_env_file(name="SEMANTIC_CONFIG_NAME_VECTOR")
    )

    USE_AZURE: bool = bool(
        True if load_params_env_file(name="USE_AZURE") in {"True", "true"} else False
    )

    USE_GPT_4: bool = bool(
        True if load_params_env_file(name="USE_GPT_4") in {"True", "true"} else False
    )

    GOOGLE_SERPER_API: str = str(load_params_env_file(name="GOOGLE_SERPER_API"))

    BLOB_ENDPOINT: str = str(load_params_env_file(name="BLOB_ENDPOINT"))
