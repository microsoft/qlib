import openai


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
