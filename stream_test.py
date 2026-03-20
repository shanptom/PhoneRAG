import json
from urllib.request import Request, urlopen

url = "http://localhost:11434/api/generate"
payload = {
    "model": "gemma3:1b",
    "prompt": "Answer in one short sentence: What is the internal codename?",
    "stream": True,
    "keep_alive": 0,
    "options": {"temperature": 0, "num_predict": 20}
}

req = Request(
    url,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)

print("Generating...\n", flush=True)

with urlopen(req) as resp:
    for raw_line in resp:
        if not raw_line.strip():
            continue
        obj = json.loads(raw_line.decode("utf-8"))
        if "response" in obj:
            print(obj["response"], end="", flush=True)

print("\n")
