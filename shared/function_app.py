# function_app.py
# Azure Function HTTP app — versão integrada a partir de generate_client_response.py
# Rotas:
#  - GET  /api/ping
#  - GET  /api/debug/env
#  - POST /api/search/obras
#
# Expect environment variables:
#  SEARCH_ENDPOINT, SEARCH_API_KEY, SEARCH_INDEX, SEARCH_API_VERSION (opt)
#  SEARCH_PROFILE (opt)
#  SEARCH_VECTOR_FIELD (opt)
#  AOAI_ENDPOINT, AOAI_API_KEY, AOAI_EMB_DEPLOYMENT, AOAI_CHAT_DEPLOYMENT, AOAI_API_VERSION (opt)
#  DEFAULT_TOPK, MAX_TOPK, REQUEST_TIMEOUT, AOAI_CHAT_MAX_TOKENS (opts)
#
# Notes:
#  - This code intentionally uses the Azure OpenAI REST endpoints (AOAI) and Azure Cognitive Search REST APIs.
#  - It prefers vectorQueries payload (same shape as generate_client_response.py) to avoid "$select annotation" issues.
#  - Minimal filesystem writes go to /tmp (Azure Functions Linux) or working dir if available.
#  - Returns a JSON payload with hits and generated answer (or fallback summary).
import os
import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import requests
import azure.functions as func

# -------------------------
# Environment helpers
# -------------------------
def get_env(name: str, default: Any = None, required: bool = False):
    v = os.environ.get(name, default)
    if required and (v is None or str(v).strip() == ""):
        raise EnvironmentError(f"Missing required environment variable: {name}")
    return v

# -------------------------
# Load configuration
# -------------------------
SEARCH_ENDPOINT = get_env("SEARCH_ENDPOINT", required=True)
SEARCH_API_KEY = get_env("SEARCH_API_KEY", required=True)
SEARCH_INDEX = get_env("SEARCH_INDEX", "kb-obras")
SEARCH_PROFILE = get_env("SEARCH_PROFILE", None)  # semantic config name (optional)
SEARCH_VECTOR_FIELD = get_env("SEARCH_VECTOR_FIELD", "content_vector")
SEARCH_API_VERSION = get_env("SEARCH_API_VERSION", "2024-07-01")

AOAI_ENDPOINT = get_env("AOAI_ENDPOINT", required=True)
AOAI_API_KEY = get_env("AOAI_API_KEY", required=True)
AOAI_EMB_DEPLOYMENT = get_env("AOAI_EMB_DEPLOYMENT", "text-embedding-3-large")
AOAI_CHAT_DEPLOYMENT = get_env("AOAI_CHAT_DEPLOYMENT", required=True)
AOAI_API_VERSION = get_env("AOAI_API_VERSION", "2024-02-15-preview")

DEFAULT_TOPK = int(get_env("DEFAULT_TOPK", "5"))
MAX_TOPK = int(get_env("MAX_TOPK", "10"))
REQUEST_TIMEOUT = int(get_env("REQUEST_TIMEOUT", "30"))
AOAI_CHAT_MAX_TOKENS = int(get_env("AOAI_CHAT_MAX_TOKENS", "800"))

HEADERS_SEARCH = {"api-key": SEARCH_API_KEY, "Content-Type": "application/json"}

# -------------------------
# Utilities
# -------------------------
def save_file(path: str, content: Any):
    try:
        with open(path, "w", encoding="utf-8") as f:
            if isinstance(content, (dict, list)):
                json.dump(content, f, ensure_ascii=False, indent=2)
            else:
                f.write(str(content))
    except Exception as e:
        logging.debug(f"Could not save file {path}: {e}")

def to_str_safe(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join([to_str_safe(x) for x in value])
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value)

# -------------------------
# AOAI: embeddings & chat
# -------------------------
def get_embedding(text: str) -> List[float]:
    url = AOAI_ENDPOINT.rstrip("/") + f"/openai/deployments/{AOAI_EMB_DEPLOYMENT}/embeddings?api-version={AOAI_API_VERSION}"
    headers = {"api-key": AOAI_API_KEY, "Content-Type": "application/json"}
    payload = {"input": text}
    resp = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"AOAI embeddings error: {resp.status_code} {resp.text}")
    body = resp.json()
    try:
        emb = body["data"][0]["embedding"]
        if not isinstance(emb, list):
            raise ValueError("Unexpected embedding format")
        return emb
    except Exception as e:
        raise RuntimeError(f"Failed to extract embedding: {e} -- resp: {json.dumps(body)[:2000]}")

def call_aoai_chat(system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    url = AOAI_ENDPOINT.rstrip("/") + f"/openai/deployments/{AOAI_CHAT_DEPLOYMENT}/chat/completions?api-version={AOAI_API_VERSION}"
    headers = {"api-key": AOAI_API_KEY, "Content-Type": "application/json"}
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": AOAI_CHAT_MAX_TOKENS,
        "temperature": 0.0
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"AOAI chat error: {resp.status_code} {resp.text}")
    return resp.json()

# -------------------------
# Azure Cognitive Search: vector query using vectorQueries (safe)
# -------------------------
def search_vector(embedding: List[float], topk: int = DEFAULT_TOPK) -> Dict[str, Any]:
    topk = min(int(topk), MAX_TOPK)
    url = SEARCH_ENDPOINT.rstrip("/") + f"/indexes/{SEARCH_INDEX}/docs/search?api-version={SEARCH_API_VERSION}"
    headers = HEADERS_SEARCH
    # avoid selecting @search.score in 'select' (causes 400 if combined with annotations)
    select = "id,doc_title,source_file,text,chunk"
    body = {
        "count": True,
        "select": select,
        "vectorQueries": [
            {
                "kind": "vector",
                "vector": embedding,
                "fields": SEARCH_VECTOR_FIELD,
                "k": topk
            }
        ],
        "top": topk
    }
    if SEARCH_PROFILE:
        body["semanticConfiguration"] = SEARCH_PROFILE
    resp = requests.post(url, headers=headers, json=body, timeout=REQUEST_TIMEOUT)
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"Azure Search error: {resp.status_code} {resp.text}")
    return resp.json()

def text_search(query: str, topk: int = DEFAULT_TOPK) -> Dict[str, Any]:
    topk = min(int(topk), MAX_TOPK)
    url = SEARCH_ENDPOINT.rstrip("/") + f"/indexes/{SEARCH_INDEX}/docs/search?api-version={SEARCH_API_VERSION}"
    headers = HEADERS_SEARCH
    select = "id,doc_title,source_file,text,chunk"
    body = {"search": query, "top": topk, "select": select}
    if SEARCH_PROFILE:
        body["semanticConfiguration"] = SEARCH_PROFILE
    resp = requests.post(url, headers=headers, json=body, timeout=REQUEST_TIMEOUT)
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"Azure Search text query failed: {resp.status_code} {resp.text}")
    return resp.json()

def get_doc_by_id(doc_id: str) -> Dict[str, Any]:
    encoded = quote(doc_id, safe='')
    url = SEARCH_ENDPOINT.rstrip("/") + f"/indexes/{SEARCH_INDEX}/docs/{encoded}?api-version={SEARCH_API_VERSION}"
    headers = {"api-key": SEARCH_API_KEY}
    resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    if resp.status_code == 200:
        return resp.json()
    return {"_fetch_error_status": resp.status_code, "_fetch_error_text": resp.text}

def dedupe_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for r in results:
        doc_id = r.get("id") or r.get("key") or r.get("raw", {}).get("id") or r.get("raw", {}).get("documentId")
        text = r.get("text") or r.get("content") or ""
        fingerprint = (doc_id if doc_id else text[:200])
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        out.append(r)
    return out

def parse_search_hits(search_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    hits = search_json.get("value") or search_json.get("results") or []
    parsed = []
    for item in hits:
        parsed.append({
            "id": item.get("id"),
            "doc_title": item.get("doc_title") or item.get("title"),
            "source_file": item.get("source_file") or item.get("source"),
            "text": item.get("text") or item.get("content"),
            "chunk": item.get("chunk"),
            "raw": item
        })
    return parsed

# -------------------------
# Prompt builder (rerank/generate) - formal/concise pt-BR (based on your original)
# -------------------------
def build_rerank_prompt(query: str, docs: List[Dict[str, Any]]) -> Tuple[str, str]:
    system = (
        "Você é um assistente profissional que consolida evidências de busca em uma resposta final destinada a clientes. "
        "Responda em Português do Brasil (pt-BR), tom formal e conciso. Produza APENAS um objeto JSON válido (sem texto adicional) "
        "com os seguintes campos obrigatórios:\n"
        " - short: resposta curta, formal, no máximo 3 frases.\n"
        " - step_by_step: lista (array) de passos claros e numerados para o usuário executar (cada passo uma string).\n"
        " - email: e-mail formal pronto para envio (string).\n"
        " - confidence: número entre 0 e 1 que representa a confiança baseada nas evidências.\n"
        " - sources: lista de objetos com {id, title, source} correspondentes às evidências utilizadas.\n"
        " - next_steps: breve sugestão de próximos passos operacionais (string).\n"
        " - call_to_action: instrução clara do que o usuário deve fazer a seguir (string).\n\n"
        "Regras estritas:\n"
        "1) NÃO invente fatos. Se uma informação não estiver nas evidências, escreva 'informação insuficiente' no campo apropriado.\n"
        "2) Cite apenas as fontes enviadas (use os ids fornecidos).\n"
        "3) Se a evidência for fraca, ajuste confidence para um valor baixo (ex.: 0.0 - 0.6). Se forte, use 0.7-1.0.\n"
        "4) Retorne APENAS o JSON (não adicione explicações fora do JSON).\n"
    )

    pieces = [f"Query: {query}", "Evidence (top results):"]
    for i, d in enumerate(docs, start=1):
        doc_id = d.get("id") or "<no-id>"
        title = d.get("doc_title") or "<no-title>"
        source = d.get("source_file") or "<no-source>"
        snippet = (d.get("text") or "")[:1000].strip().replace("\n", " ")
        pieces.append(f"{i}) id: {doc_id}\n title: {title}\n source: {source}\n snippet: {snippet}\n")
    pieces.append("Task: Com base SOMENTE nas evidências acima, gere o JSON descrito. Não inclua texto fora do JSON. Se faltar informação, use 'informação insuficiente'.")
    user = "\n\n".join(pieces)
    return system, user

def extract_chat_content(chat_resp: Dict[str, Any]) -> str:
    try:
        choices = chat_resp.get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content") or choices[0].get("text") or ""
            return content
    except Exception:
        pass
    return json.dumps(chat_resp, ensure_ascii=False)

def parse_json_from_text(text: str) -> Any:
    # Try to find the first JSON object in the text
    try:
        s = text.strip()
        if s.startswith("```"):
            parts = s.split("```")
            for p in parts:
                p = p.strip()
                if p.startswith("{") or p.startswith("["):
                    s = p
                    break
        idx = s.find("{")
        if idx != -1:
            candidate = s[idx:]
            return json.loads(candidate)
        # fallback: try to parse whole text
        return json.loads(s)
    except Exception:
        return {"text": text}

# -------------------------
# Azure Function routes
# -------------------------
app = func.FunctionApp()

@app.route(route="ping", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def ping(req: func.HttpRequest) -> func.HttpResponse:
    return func.HttpResponse(json.dumps({"status": "ok", "service": "semantic_search_obras"}, ensure_ascii=False), status_code=200, mimetype="application/json")

@app.route(route="debug/env", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def debug_env(req: func.HttpRequest) -> func.HttpResponse:
    keys = [
        "SEARCH_ENDPOINT", "SEARCH_INDEX", "SEARCH_API_KEY", "SEARCH_API_VERSION",
        "AOAI_ENDPOINT", "AOAI_EMB_DEPLOYMENT", "AOAI_CHAT_DEPLOYMENT", "AOAI_API_VERSION"
    ]
    out = {k: (os.getenv(k) or None) for k in keys}
    return func.HttpResponse(json.dumps(out, ensure_ascii=False, indent=2), status_code=200, mimetype="application/json")

def generate_client_response(body: dict) -> dict:
    """
    Adapter puro: recebe payload (dict) e retorna um dict serializável
    com a mesma estrutura do payload que hoje a rota devolve.
    """
    query = (body.get("query") or "").strip()
    if not query:
        return {"error": "missing 'query' in body", "status": 400}

    requested_topk = int(body.get("topK", DEFAULT_TOPK))
    topk = min(requested_topk, MAX_TOPK)
    debug = bool(body.get("debug", False))

    response_payload: Dict[str, Any] = {
        "query": query,
        "topK": topk,
        "used_embedding": False,
        "fallback_to_text": False,
        "hits_count": 0,
        "hits": [],
        "answer": None,
        "debug": None
    }

    # 1) Embedding + vector search (preferred)
    embedding = None
    try:
        embedding = get_embedding(query)
        response_payload["used_embedding"] = True
    except Exception as e:
        logging.warning(f"Erro gerando embedding: {e}")
        response_payload["embed_error"] = str(e)

    search_json = None
    try:
        if embedding:
            try:
                search_json = search_vector(embedding, topk=topk)
            except Exception as ve:
                logging.warning(f"Vector search failed: {ve} — falling back to text search")
                response_payload["fallback_to_text"] = True
                search_json = text_search(query, topk=topk)
        else:
            response_payload["fallback_to_text"] = True
            search_json = text_search(query, topk=topk)

        hits = parse_search_hits(search_json)
        hits = dedupe_results(hits)
        response_payload["hits_count"] = len(hits)
        response_payload["hits"] = hits
        if debug:
            response_payload["search_raw"] = search_json

    except Exception as e_search:
        logging.exception("Erro na chamada ao Azure Search")
        return {"error": f"Erro na busca: {e_search}", "status": 500}

    # 2) Try to fetch full docs for top hits (best-effort)
    full_docs = []
    for r in response_payload["hits"][:min(5, topk)]:
        doc_id = r.get("id")
        if doc_id:
            try:
                doc_full = get_doc_by_id(doc_id)
                if isinstance(doc_full, dict) and doc_full.get("id"):
                    combined = {**r, **doc_full}
                    full_docs.append(combined)
                else:
                    r["_fetch_error"] = doc_full
                    full_docs.append(r)
            except Exception as e:
                r["_fetch_error"] = str(e)
                full_docs.append(r)
        else:
            full_docs.append(r)

    # Save evidence file (optional)
    try:
        evidence_lines = []
        for i, d in enumerate(full_docs, start=1):
            evidence_lines.append(f"=== Documento #{i} ===")
            evidence_lines.append(f"id: {d.get('id')}")
            evidence_lines.append(f"title: {d.get('doc_title') or d.get('title')}")
            evidence_lines.append(f"source: {d.get('source_file') or d.get('source')}")
            evidence_lines.append(f"snippet: {(d.get('text') or '')[:2000]}")
            evidence_lines.append("\n")
        save_file("/tmp/evidence.txt", "\n".join(evidence_lines))
    except Exception:
        pass

    # 3) Reranking + generate via AOAI Chat
    try:
        system_prompt, user_prompt = build_rerank_prompt(query, full_docs[:3])
        chat_resp = call_aoai_chat(system_prompt, user_prompt)
        content = extract_chat_content(chat_resp)
        parsed = parse_json_from_text(content)
        if isinstance(parsed, dict):
            response_payload["answer"] = parsed
        else:
            response_payload["answer"] = {"text": to_str_safe(parsed)}
    except Exception as e:
        logging.warning(f"Erro na geração via AOAI Chat: {e}")
        summary_lines = []
        for idx, h in enumerate(full_docs, start=1):
            s = (h.get("text") or "").replace("\n", " ")
            one_line = (s[:300] + "...") if s else ""
            summary_lines.append(f"{idx}. Fonte: {h.get('source_file')} (chunk={h.get('chunk')})\n{one_line}")
        fallback_answer = "Resumo dos principais trechos encontrados:\n\n" + "\n\n".join(summary_lines)
        fallback_answer += "\n\nRecomendações: 1) Abrir chamado no Portal de Atendimento e anexar evidências. 2) Informar unidade/setor e nível de prioridade."
        response_payload["answer"] = {"text": fallback_answer}
        response_payload["gen_error"] = str(e)

    if debug:
        response_payload["debug"] = {
            "embed_used": response_payload.get("used_embedding"),
            "fallback_to_text": response_payload.get("fallback_to_text"),
            "embed_error": response_payload.get("embed_error"),
        }

    return response_payload


@app.route(route="search/obras", methods=["POST"], auth_level=func.AuthLevel.ANONYMOUS)
def semantic_search_obras(req: func.HttpRequest) -> func.HttpResponse:
    try:
        body = req.get_json() if req.get_body() else {}
    except Exception:
        body = {}

    query = (body.get("query") or "").strip()
    if not query:
        return func.HttpResponse(json.dumps({"error": "missing 'query' in body"}, ensure_ascii=False), status_code=400, mimetype="application/json")

    requested_topk = int(body.get("topK", DEFAULT_TOPK))
    topk = min(requested_topk, MAX_TOPK)
    debug = bool(body.get("debug", False))

    response_payload: Dict[str, Any] = {
        "query": query,
        "topK": topk,
        "used_embedding": False,
        "fallback_to_text": False,
        "hits_count": 0,
        "hits": [],
        "answer": None,
        "debug": None
    }

    # 1) Embedding + vector search (preferred)
    embedding = None
    try:
        embedding = get_embedding(query)
        response_payload["used_embedding"] = True
    except Exception as e:
        logging.warning(f"Erro gerando embedding: {e}")
        response_payload["embed_error"] = str(e)

    search_json = None
    try:
        if embedding:
            try:
                search_json = search_vector(embedding, topk=topk)
            except Exception as ve:
                logging.warning(f"Vector search failed: {ve} — falling back to text search")
                response_payload["fallback_to_text"] = True
                search_json = text_search(query, topk=topk)
        else:
            response_payload["fallback_to_text"] = True
            search_json = text_search(query, topk=topk)

        hits = parse_search_hits(search_json)
        hits = dedupe_results(hits)
        response_payload["hits_count"] = len(hits)
        response_payload["hits"] = hits
        if debug:
            response_payload["search_raw"] = search_json

    except Exception as e_search:
        logging.exception("Erro na chamada ao Azure Search")
        return func.HttpResponse(json.dumps({"error": f"Erro na busca: {e_search}"}, ensure_ascii=False), status_code=500, mimetype="application/json")

    # 2) Try to fetch full docs for top hits (best-effort)
    full_docs = []
    for r in response_payload["hits"][:min(5, topk)]:
        doc_id = r.get("id")
        if doc_id:
            try:
                doc_full = get_doc_by_id(doc_id)
                if isinstance(doc_full, dict) and doc_full.get("id"):
                    combined = {**r, **doc_full}
                    full_docs.append(combined)
                else:
                    r["_fetch_error"] = doc_full
                    full_docs.append(r)
            except Exception as e:
                r["_fetch_error"] = str(e)
                full_docs.append(r)
        else:
            full_docs.append(r)

    # Save evidence file (optional, non-critical)
    try:
        evidence_lines = []
        for i, d in enumerate(full_docs, start=1):
            evidence_lines.append(f"=== Documento #{i} ===")
            evidence_lines.append(f"id: {d.get('id')}")
            evidence_lines.append(f"title: {d.get('doc_title') or d.get('title')}")
            evidence_lines.append(f"source: {d.get('source_file') or d.get('source')}")
            evidence_lines.append(f"snippet: {(d.get('text') or '')[:2000]}")
            evidence_lines.append("\n")
        save_file("/tmp/evidence.txt", "\n".join(evidence_lines))
    except Exception:
        pass

    # 3) Reranking + generate via AOAI Chat
    try:
        system_prompt, user_prompt = build_rerank_prompt(query, full_docs[:3])
        chat_resp = call_aoai_chat(system_prompt, user_prompt)
        content = extract_chat_content(chat_resp)
        parsed = parse_json_from_text(content)
        # ensure result is dict with expected fields; otherwise fallback to raw text
        if isinstance(parsed, dict):
            response_payload["answer"] = parsed
        else:
            response_payload["answer"] = {"text": to_str_safe(parsed)}
    except Exception as e:
        logging.warning(f"Erro na geração via AOAI Chat: {e}")
        # fallback: build a quick summary
        summary_lines = []
        for idx, h in enumerate(full_docs, start=1):
            s = (h.get("text") or "").replace("\n", " ")
            one_line = (s[:300] + "...") if s else ""
            summary_lines.append(f"{idx}. Fonte: {h.get('source_file')} (chunk={h.get('chunk')})\n{one_line}")
        fallback_answer = "Resumo dos principais trechos encontrados:\n\n" + "\n\n".join(summary_lines)
        # some pragmatic recommendations
        fallback_answer += "\n\nRecomendações: 1) Abrir chamado no Portal de Atendimento e anexar evidências. 2) Informar unidade/setor e nível de prioridade."
        response_payload["answer"] = {"text": fallback_answer}
        response_payload["gen_error"] = str(e)

    if debug:
        response_payload["debug"] = {
            "embed_used": response_payload.get("used_embedding"),
            "fallback_to_text": response_payload.get("fallback_to_text"),
            "embed_error": response_payload.get("embed_error"),
        }

    return func.HttpResponse(json.dumps(response_payload, ensure_ascii=False, indent=2), status_code=200, mimetype="application/json")
