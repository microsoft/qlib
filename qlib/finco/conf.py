# TODO: use pydantic for other modules in Qlib
from pydantic import (BaseSettings)
from qlib.finco.utils import Singleton

import os

class Config(Singleton):
    def __init__(self):
        self.use_azure = os.getenv("USE_AZURE") == "True"
        self.temperature = 0.5 if os.getenv("TEMPERATURE") is None else float(os.getenv("TEMPERATURE"))
        self.max_tokens = 800 if os.getenv("MAX_TOKENS") is None else int(os.getenv("MAX_TOKENS"))
        
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.use_azure = os.getenv("USE_AZURE") == "True"
        self.azure_api_base = os.getenv("AZURE_API_BASE")
        self.azure_api_version = os.getenv("AZURE_API_VERSION")
        self.model = os.getenv("MODEL") or ("gpt-35-turbo" if self.use_azure else "gpt-3.5-turbo")

        self.max_retry = int(os.getenv("MAX_RETRY")) if os.getenv("MAX_RETRY") is not None else None

        self.continous_mode = os.getenv("CONTINOUS_MODE") == "True" if os.getenv("CONTINOUS_MODE") is not None else False
        self.debug_mode = os.getenv("DEBUG_MODE") == "True" if os.getenv("DEBUG_MODE") is not None else False