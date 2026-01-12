import os, json
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

def main():
    assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not set"

    client = OpenAI()
    candidates = ["Drill_8mm"]
    query = "8mm drilling tool"

    schema = {
        "type": "object",
        "properties": {
            "tool_id": {
                "anyOf": [
                    {"type": "string", "enum": candidates},
                    {"type": "null"},
                ]
            }
        },
        "required": ["tool_id"],
        "additionalProperties": False,
    }

    resp = client.responses.create(
        model="gpt-5-mini",
        input=[
            {"role": "system", "content": "Pick the best tool_id from Candidates. If none match, return null."},
            {"role": "user", "content": f"Query: {query}\nCandidates: {candidates}"},
        ],
        text={"format": {"type": "json_schema", "name": "tool_match", "strict": True, "schema": schema}},
    )

    print("RAW output_text:", resp.output_text)
    data = json.loads(resp.output_text)
    print("PARSED tool_id:", data.get("tool_id"))

if __name__ == "__main__":
    main()
