import json

class Singleton():
    _instance = None
    def __new__(cls, *args, **kwargs):  
        if cls._instance is None:  
            cls._instance = super().__new__(cls, *args, **kwargs)  
        return cls._instance  

def parse_json(response):
    try:
        return json.loads(response)
    except json.decoder.JSONDecodeError:
        pass

    raise Exception(f"Failed to parse response: {response}, please report it or help us to fix it.")
