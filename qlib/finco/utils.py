import json

def parse_json(response):
    try:
        return json.loads(response)
    except json.decoder.JSONDecodeError:
        pass

    raise Exception(f"Failed to parse response: {response}, please report it or help us to fix it.")