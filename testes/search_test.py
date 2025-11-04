#!/usr/bin/env python3
"""
consulta_vector_search.py

Script completo pronto para copiar/colar.

O fluxo:
  1. Recebe uma query de texto (argumento CLI ou input interativo)
  2. Gera embedding via Azure OpenAI (AOAI) usando deployment de embeddings
  3. Executa uma busca vetorial no Azure Cognitive Search (index + field vector)
  4. Exibe resultados (id, título, source e snippet)

Uso:
  python consulta_vector_search.py "Qual é o status da obra X?"
  ou
  python consulta_vector_search.py
  (então será solicitado digitar a query)

OBS:
  - Este script usa variáveis de ambiente (conforme seu arquivo JSON):
      SEARCH_ENDPOINT, SEARCH_API_KEY, SEARCH_INDEX, SEARCH_PROFILE,
      SEARCH_VECTOR_FIELD, SEARCH_API_VERSION,
      AOAI_ENDPOINT, AOAI_API_KEY, AOAI_EMB_DEPLOYMENT, AOAI_API_VERSION,
      DEFAULT_TOPK, REQUEST_TIMEOUT
  - Certifique-se de carregar essas variáveis no ambiente antes de rodar
    (ou exportá-las em Windows PowerShell / CMD).
"""

import os
import sys
import json
import requests
from typing import List, Any

# -------------------------
# Helpers para environment
# -------------------------
def get_env(name: str, default: Any = None, required: bool = False):
    v = os.environ.get(name, default)
    if required and (v is None or str(v) == ""):
        raise EnvironmentError(f"Missing required environment variable: {name}")
    return v

# -------------------------
# Leitura das variáveis
# -------------------------
SEARCH_ENDPOINT = get_env("SEARCH_ENDPOINT", required=True)                  # ex: https://see-h-ai-crm-searchbot.search.windows.net
SEARCH_API_KEY = get_env("SEARCH_API_KEY", required=True)
SEARCH_INDEX = get_env("SEARCH_INDEX", "kb-obras")
SEARCH_PROFILE = get_env("SEARCH_PROFILE", "vprofile")
SEARCH_VECTOR_FIELD = get_env("SEARCH_VECTOR_FIELD", "content_vector")
SEARCH_API_VERSION = get_env("SEARCH_API_VERSION", "2024-07-01")

AOAI_ENDPOINT = get_env("AOAI_ENDPOINT", required=True)                     # ex: https://seducouvidoriacrm-dev.openai.azure.com
AOAI_API_KEY = get_env("AOAI_API_KEY", required=True)
AOAI_EMB_DEPLOYMENT = get_env("AOAI_EMB_DEPLOYMENT", "text-embedding-3-large")
AOAI_API_VERSION = get_env("AOAI_API_VERSION", "2024-02-15-preview")

DEFAULT_TOPK = int(get_env("DEFAULT_TOPK", "5"))
MAX_TOPK = int(get_env("MAX_TOPK", "10"))
REQUEST_TIMEOUT = int(get_env("REQUEST_TIMEOUT", "30"))

# -------------------------
# Funções principais
# -------------------------
def get_embedding(text: str) -> List[float]:
    """
    Gera embedding via Azure OpenAI (AOAI).
    Retorna lista de floats.
    """
    url = AOAI_ENDPOINT.rstrip("/") + f"/openai/deployments/{AOAI_EMB_DEPLOYMENT}/embeddings?api-version={AOAI_API_VERSION}"
    headers = {
        "api-key": AOAI_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {"input": text}
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
    except Exception as e:
        raise RuntimeError(f"Erro ao chamar AOAI embeddings: {e}")

    if resp.status_code not in (200, 201):
        raise RuntimeError(f"AOAI embeddings error: {resp.status_code} {resp.text}")

    body = resp.json()
    # padrão: body["data"][0]["embedding"]
    try:
        emb = body["data"][0]["embedding"]
        if not isinstance(emb, list):
            raise ValueError("Formato inesperado do embedding")
        return emb
    except Exception as e:
        raise RuntimeError(f"Não foi possível extrair embedding da resposta AOAI: {e} -- full response: {json.dumps(body)[:1000]}")

def search_vector(embedding: List[float], topk: int = DEFAULT_TOPK):
    """
    Executa a busca vetorial no Azure Cognitive Search (REST API).
    Retorna o JSON da resposta.
    """
    topk = min(int(topk), MAX_TOPK)
    url = SEARCH_ENDPOINT.rstrip("/") + f"/indexes/{SEARCH_INDEX}/docs/search?api-version={SEARCH_API_VERSION}"
    headers = {
        "api-key": SEARCH_API_KEY,
        "Content-Type": "application/json"
    }

    search_body = {
        "count": True,
        # Ajuste os campos do 'select' conforme seu index (aqui vamos imprimir id, doc_title, source_file, text)
        "select": "id,doc_title,source_file,text",
        "vectorQueries": [
            {
                "kind": "vector",
                "vector": embedding,
                "fields": SEARCH_VECTOR_FIELD,
                "k": topk
            }
        ],
        "top": topk,
        "semanticConfiguration": SEARCH_PROFILE
    }

    try:
        resp = requests.post(url, headers=headers, json=search_body, timeout=REQUEST_TIMEOUT)
    except Exception as e:
        raise RuntimeError(f"Erro ao chamar Azure Cognitive Search: {e}")

    if resp.status_code not in (200, 201):
        raise RuntimeError(f"Azure Search error: {resp.status_code} {resp.text}")

    return resp.json()

def pretty_print_results(res_json: dict):
    """
    Formata e imprime os resultados no terminal.
    """
    total_count = res_json.get("@odata.count", None) or res_json.get("count", None)
    values = res_json.get("value", []) or res_json.get("results", res_json)

    print("\n=== Resultados ===")
    if total_count is not None:
        print(f"Total estimado: {total_count}")
    print(f"Registros retornados: {len(values)}\n")

    for idx, doc in enumerate(values, start=1):
        # Alguns campos do Azure vem com prefixo @search.* (score); tentamos exibir alguns deles
        score = doc.get("@search.score") or doc.get("@search.rerankerScore") or doc.get("score")
        doc_id = doc.get("id") or doc.get("doc_id") or doc.get("documentId") or "<sem id>"
        title = doc.get("doc_title") or doc.get("title") or "<sem título>"
        source = doc.get("source_file") or doc.get("source") or "<sem source>"
        text = doc.get("text") or doc.get("content") or ""
        # curto snippet (primeira linha / 300 chars)
        snippet = text.replace("\n", " ").strip()[:300] + ("..." if len(text) > 300 else "")

        print(f"[{idx}] id: {doc_id}  score: {score}")
        print(f"     título: {title}")
        print(f"     source: {source}")
        print(f"     snippet: {snippet}\n")

# -------------------------
# Main / CLI
# -------------------------
def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:]).strip()
    else:
        query = input("Digite a sua pergunta/consulta para gerar embedding e consultar o index: ").strip()

    if not query:
        print("Query vazia. Abortando.")
        sys.exit(1)

    print(f"\nGerando embedding para a query: {query!r} (via AOAI '{AOAI_EMB_DEPLOYMENT}')")
    embedding = get_embedding(query)
    print(f"Embedding gerado (tamanho={len(embedding)}). Executando busca vetorial no index '{SEARCH_INDEX}'...")

    res = search_vector(embedding, topk=DEFAULT_TOPK)
    pretty_print_results(res)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERRO] {e}")
        sys.exit(1)
