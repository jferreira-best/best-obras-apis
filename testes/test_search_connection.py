import os, json, requests
AOAI_ENDPOINT = os.environ.get("AOAI_ENDPOINT")
AOAI_API_KEY = os.environ.get("AOAI_API_KEY")
DEPLOY = os.environ.get("AOAI_EMB_DEPLOYMENT")
url = f"{AOAI_ENDPOINT}/openai/deployments/{DEPLOY}/embeddings?api-version=2024-02-15-preview"
hdr = {"api-key": AOAI_API_KEY, "Content-Type":"application/json"}
r = requests.post(url, headers=hdr, json={"input":"teste para dimensão"})
print("status", r.status_code)
print(r.json())
print("len:", len(r.json()["data"][0]["embedding"]))
