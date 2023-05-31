import json

def params_from_json(p: str) -> dict:
    params = None
    with open(p, 'r') as f:
        params = json.load(f)

    return params