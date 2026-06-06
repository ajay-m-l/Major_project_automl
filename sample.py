import requests

res = requests.post(
    "http://127.0.0.1:11434/api/generate",
    json={
        "model": "llama3",
        "prompt": "Hello",
        "stream": False
    }
)

print(res.json())