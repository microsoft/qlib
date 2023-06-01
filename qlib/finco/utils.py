import json
from qlib.finco.llm import try_create_chat_completion
from qlib.finco.conf import Config
from qlib.log import get_module_logger
from pathlib import Path


def parse_json(response):
    try:
        return json.loads(response)
    except json.decoder.JSONDecodeError:
        pass

    raise Exception(f"Failed to parse response: {response}, please report it or help us to fix it.")

def build_messages_and_create_chat_completion(user_prompt, system_prompt=None):
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
    response = try_create_chat_completion(messages=messages)
    return response
