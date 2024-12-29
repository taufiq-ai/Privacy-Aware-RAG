import json


def content_to_json(content: str) -> dict:
    clean = content.replace("```", "").replace("\n", "").replace("json", "")
    return json.loads(clean)