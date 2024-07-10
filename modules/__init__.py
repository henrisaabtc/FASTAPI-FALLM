"""init module for sharing ressource"""

import os

import certifi

import logging

from modules.token_counter import TokenCounter

os.environ["SSL_CERT_FILE"] = certifi.where()


class LoggerCustomAzure:
    def info(self, msg, *args, **kwargs):
        logging.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        logging.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        logging.error(msg, *args, **kwargs)


logging.getLogger().setLevel(logging.INFO)

logger = LoggerCustomAzure()

token_counter = TokenCounter(token=0)
