import time
import openai
from typing import Optional
from qlib.finco.conf import Config


def example():
    response = openai.ChatCompletion.create(
        engine="gpt-35-turbo",  # The deployment name you chose when you deployed the ChatGPT or GPT-4 model.
        # engine="gpt-4",  # NOTE: this raises this error: openai.error.RateLimitError: Requests to the Creates a completion for the chat message Operation under Azure OpenAI API version 2023-05-15 have exceeded call rate limit of your current OpenAI S0 pricing tier
        # engine="gpt-4-32k",  # This works for only;
        messages=[
            {"role": "system", "content": "Assistant is a large language model trained by OpenAI."},
            {"role": "user", "content": "Who were the founders of Microsoft?"},
        ],
    )
    print(response)

def try_create_chat_completion(max_retry=10, **kwargs):
    cfg = Config()
    max_retry = cfg.max_retry if cfg.max_retry is not None else max_retry
    for i in range(max_retry):
        try:
            response = create_chat_completion(**kwargs)
            return response
        except openai.error.RateLimitError as e:
            print(e)
            print(f"Retrying {i+1}th time...")
            time.sleep(1)
            continue
    raise Exception(f"Failed to create chat completion after {max_retry} retries.")

def create_chat_completion(
    messages,
    model = None,
    temperature: float = None,
    max_tokens: Optional[int] = None,
) -> str:
    cfg = Config()

    if temperature is None:
        temperature = cfg.temperature
    if max_tokens is None:
        max_tokens = cfg.max_tokens
    
    openai.api_key = cfg.openai_api_key
    if cfg.use_azure:
        openai.api_type = "azure"
        openai.api_base = cfg.azure_api_base
        openai.api_version = cfg.azure_api_version
        response = openai.ChatCompletion.create(
            engine=cfg.model,
            messages=messages,
            max_tokens=cfg.max_tokens,
        )
    else:
        response = openai.ChatCompletion.create(
            model=cfg.model,
            messages=messages,
        )
    resp = response.choices[0].message["content"]
    return resp

if __name__ == "__main__":
    create_chat_completion()