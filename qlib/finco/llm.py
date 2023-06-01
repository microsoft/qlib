import os
import time
import openai
import json
from typing import Optional
from qlib.log import get_module_logger
from qlib.finco.conf import Config
from qlib.finco.utils import Singleton


class APIBackend(Singleton):
    def __init__(self):
        self.cfg = Config()
        openai.api_key = self.cfg.openai_api_key
        if self.cfg.use_azure:
            openai.api_type = "azure"
            openai.api_base = self.cfg.azure_api_base
            openai.api_version = self.cfg.azure_api_version
        self.use_azure = self.cfg.use_azure

        self.debug_mode = False
        if self.cfg.debug_mode:
            self.debug_mode = True
            cwd = os.getcwd()
            self.cache_file_location = os.path.join(cwd, "prompt_cache.json")
            self.cache = json.load(open(self.cache_file_location, "r")) if os.path.exists(self.cache_file_location) else {}
    
    def build_messages_and_create_chat_completion(self, user_prompt, system_prompt=None):
        """build the messages to avoid implementing several redundant lines of code"""
        cfg = Config()
        # TODO: system prompt should always be provided. In development stage we can use default value
        if system_prompt is None:
            try:
                system_prompt = cfg.system_prompt
            except AttributeError:
                get_module_logger("finco").warning("system_prompt is not set, using default value.")
                system_prompt = "You are an AI assistant who helps to answer user's questions about finance."
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]
        response = self.try_create_chat_completion(messages=messages)
        return response

    def try_create_chat_completion(self, max_retry=10, **kwargs):
        max_retry = self.cfg.max_retry if self.cfg.max_retry is not None else max_retry
        for i in range(max_retry):
            try:
                response = self.create_chat_completion(**kwargs)
                return response
            except openai.error.RateLimitError as e:
                print(e)
                print(f"Retrying {i+1}th time...")
                time.sleep(1)
                continue
        raise Exception(f"Failed to create chat completion after {max_retry} retries.")

    def create_chat_completion(
        self,
        messages,
        model = None,
        temperature: float = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        
        if self.debug_mode:
            if messages[1]["content"] in self.cache:
                return self.cache[messages[1]["content"]]

        if temperature is None:
            temperature = self.cfg.temperature
        if max_tokens is None:
            max_tokens = self.cfg.max_tokens
        
        if self.cfg.use_azure:
            response = openai.ChatCompletion.create(
                engine=self.cfg.model,
                messages=messages,
                max_tokens=self.cfg.max_tokens,
            )
        else:
            response = openai.ChatCompletion.create(
                model=self.cfg.model,
                messages=messages,
            )
        resp = response.choices[0].message["content"]
        if self.debug_mode:
            self.cache[messages[1]["content"]] = resp
            json.dump(self.cache, open(self.cache_file_location, "w"))
        return resp
