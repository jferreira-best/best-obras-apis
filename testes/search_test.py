# test_openai_envs.py
import os
import requests
import json
import logging
from textwrap import shorten

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Variáveis que vamos checar (adapte se usa nomes diferentes)
env_vars = [
    "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_KEY", "AZURE_OPENAI_DEPLOYMENT",
    "OPENAI_API_KEY", "OPENAI_API_BASE", "OPENAI_API_TYPE", "OPENAI_API_VERSION",
    "EMBEDDING_DEPLOYMENT", "OPENAI_ENDPOINT", "OPENAI_KEY", "OPENAI_DEPLOYMENT"
]

def show_envs():
    logging.info("== Verificando variáveis de ambiente == ")
    for v in env_vars:
        val = os.getenv(v)
        if val:
            print(f"{v} = {shorten(val, width=120, placeholder='...[trunc]')}")
        else:
            print(f"{v} = <NÂO DEFINIDA>")

def test_azure_deployments(endpoint, key, api_version="2023-10-01"):
    if not endpoint or not key:
        logging.warning("Azure endpoint/key não definidos — pulando teste de deployments.")
        return
    url = endpoint.rstrip("/") + f"/openai/deployments?api-version={api_version}"
    logging.info("GET deployments -> %s", url)
    try:
        resp = requests.get(url, headers={"api-key": key}, timeout=15)
        logging.info("Status: %s", resp.status_code)
        text = resp.text or "<empty body>"
        print("Body (trecho):", shorten(text, width=200, placeholder="...[trunc]"))
    except Exception as e:
        logging.exception("Erro ao chamar /deployments")

def test_azure_chat(endpoint, key, deployment, api_version="2023-10-01"):
    if not (endpoint and key and deployment):
        logging.warning("Faltando endpoint/key/deployment para teste de chat Azure — pulando.")
        return
    url = endpoint.rstrip("/") + f"/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
    payload = {
        "messages": [
            {"role": "system", "content": "Você é um teste rápido. Responda com 'ok'."},
            {"role": "user", "content": "teste"}
        ],
        "max_tokens": 10
    }
    logging.info("POST chat -> %s", url)
    try:
        resp = requests.post(url, headers={"api-key": key, "Content-Type": "application/json"},
                             json=payload, timeout=20)
        logging.info("Status: %s", resp.status_code)
        print("Resposta (trecho):", shorten(resp.text or "<empty>", width=400, placeholder="...[trunc]"))
        # tente parsear JSON com tratamento para resposta vazia/HTML
        try:
            j = resp.json()
            print("JSON keys:", list(j.keys()) if isinstance(j, dict) else type(j))
        except ValueError:
            logging.error("Resposta não é JSON — cuidado ao chamar .json() no seu código.")
    except Exception:
        logging.exception("Erro no POST chat Azure")

def test_openai_models(api_key):
    if not api_key:
        logging.warning("OPENAI_API_KEY não definida — pulando teste OpenAI padrão.")
        return
    url = "https://api.openai.com/v1/models"
    logging.info("GET OpenAI models -> %s", url)
    try:
        resp = requests.get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=15)
        logging.info("Status: %s", resp.status_code)
        print("Resposta (trecho):", shorten(resp.text or "<empty>", width=400, placeholder="...[trunc]"))
    except Exception:
        logging.exception("Erro ao chamar OpenAI public API")

if __name__ == "__main__":
    show_envs()
    # Tente com os nomes mais prováveis (adapte se usa outros)
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("OPENAI_ENDPOINT") or os.getenv("OPENAI_API_BASE")
    azure_key = os.getenv("AZURE_OPENAI_KEY") or os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
    azure_deployment = (os.getenv("AZURE_OPENAI_DEPLOYMENT") or os.getenv("OPENAI_DEPLOYMENT")
                        or os.getenv("OPENAI_DEPLOYMENT_NAME") or os.getenv("EMBEDDING_DEPLOYMENT"))
    openai_api_key = os.getenv("OPENAI_API_KEY")

    print("\n== Teste Azure OpenAI: listar deployments ==")
    test_azure_deployments(azure_endpoint, azure_key, api_version=os.getenv("OPENAI_API_VERSION", "2023-10-01"))

    print("\n== Teste Azure OpenAI: chamada de chat para deployment informado ==")
    test_azure_chat(azure_endpoint, azure_key, azure_deployment, api_version=os.getenv("OPENAI_API_VERSION", "2023-10-01"))

    print("\n== Teste OpenAI público: listar modelos (se tiver OPENAI_API_KEY) ==")
    test_openai_models(openai_api_key)

    print("\n== FIM dos testes ==")
