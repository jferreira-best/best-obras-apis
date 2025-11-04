# shared/function_app.py
"""
Shared function app helpers and main handler used by Azure Functions.
Updated: stronger AOAI prompt (returns structured JSON), better fallbacks and logging.
"""
from __future__ import annotations
import os
import json
import logging
import re
import textwrap
from typing import Any, Dict, List, Optional, Tuple
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from json.decoder import JSONDecoder

# ---------------------------
# Configuration (env / defaults)
# ---------------------------
COG_SEARCH_ENDPOINT = os.environ.get("COG_SEARCH_ENDPOINT") or os.environ.get("SEARCH_ENDPOINT")
COG_SEARCH_KEY      = os.environ.get("COG_SEARCH_KEY") or os.environ.get("SEARCH_API_KEY")
COG_SEARCH_INDEX    = os.environ.get("COG_SEARCH_INDEX") or os.environ.get("SEARCH_INDEX")

AOAI_ENDPOINT = os.environ.get("AOAI_ENDPOINT") or os.environ.get("OPENAI_ENDPOINT") or ""
AOAI_API_KEY  = os.environ.get("AOAI_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""

AOAI_ENDPOINT = AOAI_ENDPOINT.rstrip("/")
AOAI_EMB_DEPLOYMENT = os.environ.get("AOAI_EMB_DEPLOYMENT") or os.environ.get("AOAI_EMBEDDING_DEPLOYMENT") or "text-embedding-3-large"
AOAI_CHAT_DEPLOYMENT = os.environ.get("AOAI_CHAT_DEPLOYMENT") or os.environ.get("AOAI_DEPLOYMENT") or "gpt-4o-mini"
AOAI_API_VERSION = os.environ.get("AOAI_API_VERSION") or "2024-02-15-preview"
AOAI_CHAT_MAX_TOKENS = int(os.environ.get("AOAI_CHAT_MAX_TOKENS", "1024"))

COG_SEARCH_API_VERSION = os.environ.get("SEARCH_API_VERSION") or os.environ.get("COG_SEARCH_API_VERSION") or "2024-07-01"
COG_SEARCH_ENDPOINT = (COG_SEARCH_ENDPOINT or "").rstrip("/")
COG_SEARCH_KEY = COG_SEARCH_KEY or ""
COG_SEARCH_INDEX = COG_SEARCH_INDEX or os.environ.get("SEARCH_INDEX") or ""

REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "30"))
DEFAULT_TOPK = int(os.environ.get("DEFAULT_TOPK", "5"))
MAX_TOPK = int(os.environ.get("MAX_TOPK", "15"))

COG_SEARCH_VECTOR_FIELD = os.environ.get("SEARCH_VECTOR_FIELD") or os.environ.get("COG_SEARCH_VECTOR_FIELD") or "content_vector"
SEARCH_SELECT_FIELDS = os.environ.get("SEARCH_SELECT_FIELDS") or "id,text,doc_title,source_file"

FORCE_TEXT_SEARCH = str(os.environ.get("FORCE_TEXT_SEARCH") or "false").lower() in ("1", "true", "yes")

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------
# HTTP Session with Retry
# ---------------------------
SESSION = requests.Session()
_RETRY = Retry(
    total=3,
    backoff_factor=0.6,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=frozenset(["GET", "POST", "PUT", "DELETE", "HEAD"])
)
SESSION.mount("https://", HTTPAdapter(max_retries=_RETRY))
SESSION.mount("http://", HTTPAdapter(max_retries=_RETRY))


# ---------------------------
# Utility helpers
# ---------------------------
def to_str_safe(v: Any, max_len: int = 2000) -> str:
    try:
        s = str(v)
    except Exception:
        s = "<unserializable>"
    return s if len(s) <= max_len else s[:max_len] + "..."


def request_json(method: str, url: str, headers: Optional[Dict[str, str]] = None,
                 json_payload: Optional[Dict[str, Any]] = None,
                 params: Optional[Dict[str, Any]] = None,
                 timeout: int = REQUEST_TIMEOUT) -> Any:
    headers = headers or {}
    try:
        if method.lower() == "post":
            resp = SESSION.post(url, headers=headers, json=json_payload, params=params, timeout=timeout)
        else:
            resp = SESSION.get(url, headers=headers, params=params, timeout=timeout)
    except Exception as e:
        logger.exception("HTTP request failed")
        raise RuntimeError(f"HTTP request failed: {e}")

    if resp.status_code not in (200, 201):
        snippet = (resp.text or "")[:2000]
        raise RuntimeError(f"HTTP {resp.status_code} for {url}: {snippet}")

    try:
        return resp.json()
    except Exception:
        return resp.text


def parse_json_from_text(text: str) -> Any:
    if not text:
        return None
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)

    decoder = JSONDecoder()
    for i in range(len(s)):
        try:
            obj, idx = decoder.raw_decode(s[i:])
            return obj
        except ValueError:
            continue
    try:
        return json.loads(s)
    except Exception:
        return {"text": s}


# ---------------------------
# Synthesizer (fallback summarizer)
# ---------------------------
def synthesize_answer(query: str, hits: list, max_total_chars: int = 3500) -> str:
    # If AOAI not configured, just concatenate short excerpts
    aoai_endpoint = AOAI_ENDPOINT
    aoai_key = AOAI_API_KEY
    aoai_deploy = AOAI_CHAT_DEPLOYMENT
    aoai_api_ver = AOAI_API_VERSION or "2024-02-15-preview"

    if not (aoai_endpoint and aoai_key and aoai_deploy):
        texts = []
        for h in hits[:5]:
            t = (h.get("text") or h.get("raw", {}).get("text", "") or "").strip()
            texts.append(textwrap.shorten(t, width=800, placeholder=" ..."))
        return "\n\n".join(texts) or "No content to summarize."

    snippets = []
    total_chars = 0
    for i, h in enumerate(hits, start=1):
        raw_text = (h.get("text") or h.get("raw", {}).get("text", "") or "").strip()
        src = h.get("raw", {}).get("source_file") or h.get("source_file") or h.get("id", "unknown")
        remaining = max_total_chars - total_chars
        if remaining <= 0:
            break
        per_hit = min(800, remaining)
        excerpt = textwrap.shorten(raw_text.replace("\n", " "), width=per_hit, placeholder=" ...")
        snippet = f"[{i}] SOURCE: {src}\n{excerpt}"
        snippets.append(snippet)
        total_chars += len(snippet)

    context_text = "\n\n".join(snippets) if snippets else "No document content available."

    system_msg = (
        "You are an assistant that reads short document excerpts and the user's question, "
        "then returns a single concise, actionable answer in Portuguese. "
        "Be precise, give 3-6 short action steps or recommendations, and cite the source index in square brackets like [1]."
    )
    user_msg = f"Pergunta: {query}\n\nTrechos:\n{context_text}\n\nGere uma resposta curta e prática (3-6 bullets) em Português."

    payload = {
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        "temperature": 0.0,
        "max_tokens": 512
    }

    url = aoai_endpoint.rstrip("/") + f"/openai/deployments/{aoai_deploy}/chat/completions?api-version={aoai_api_ver}"
    headers = {"api-key": aoai_key, "Content-Type": "application/json"}

    try:
        resp = request_json("post", url, headers=headers, json_payload=payload)
        choices = resp.get("choices") or []
        if choices and isinstance(choices, list):
            first = choices[0]
            msg = first.get("message", {}).get("content") or first.get("text") or ""
        else:
            msg = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
        return msg.strip() if isinstance(msg, str) else json.dumps(msg)
    except Exception as e:
        fallback_texts = []
        for i, h in enumerate(hits[:5], start=1):
            t = (h.get("text") or h.get("raw", {}).get("text", "") or "").strip()
            fallback_texts.append(f"[{i}] " + textwrap.shorten(t, width=400, placeholder=" ..."))
        return "Não foi possível sintetizar via AOAI: " + str(e) + "\n\nFragmentos:\n\n" + "\n\n".join(fallback_texts)


# ---------------------------
# AOAI helpers
# ---------------------------
def get_embedding(text: str) -> List[float]:
    if not AOAI_ENDPOINT or not AOAI_API_KEY:
        raise RuntimeError("AOAI_ENDPOINT or AOAI_API_KEY not set in environment")
    url = f"{AOAI_ENDPOINT}/openai/deployments/{AOAI_EMB_DEPLOYMENT}/embeddings?api-version={AOAI_API_VERSION}"
    headers = {"api-key": AOAI_API_KEY, "Content-Type": "application/json"}
    payload = {"input": text}
    body = request_json("post", url, headers=headers, json_payload=payload)
    try:
        emb = body["data"][0]["embedding"]
        if not isinstance(emb, list):
            raise RuntimeError("Unexpected embedding format")
        return emb
    except Exception as e:
        logger.exception("Failed to extract embedding from AOAI response")
        raise RuntimeError(f"Failed to extract embedding: {e} -- snippet: {to_str_safe(body)[:1000]}")


def call_aoai_chat(system_prompt: str, user_prompt: str, max_tokens: int = None) -> Any:
    if not AOAI_ENDPOINT or not AOAI_API_KEY:
        raise RuntimeError("AOAI_ENDPOINT or AOAI_API_KEY not set")
    if max_tokens is None:
        max_tokens = AOAI_CHAT_MAX_TOKENS
    url = f"{AOAI_ENDPOINT}/openai/deployments/{AOAI_CHAT_DEPLOYMENT}/chat/completions?api-version={AOAI_API_VERSION}"
    headers = {"api-key": AOAI_API_KEY, "Content-Type": "application/json"}
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0
    }
    return request_json("post", url, headers=headers, json_payload=payload)


def extract_chat_content(chat_resp: Any) -> str:
    if not chat_resp:
        return ""
    try:
        if isinstance(chat_resp, dict):
            choices = chat_resp.get("choices") or chat_resp.get("choices", [])
            if choices and isinstance(choices, list):
                c0 = choices[0]
                if isinstance(c0, dict):
                    msg = c0.get("message") or c0.get("message", {})
                    if isinstance(msg, dict) and "content" in msg:
                        return msg["content"]
                    if "text" in c0:
                        return c0["text"]
        return to_str_safe(chat_resp)
    except Exception:
        logger.exception("Failed to extract chat content")
        return to_str_safe(chat_resp)


# ---------------------------
# Azure Cognitive Search helpers
# ---------------------------
def search_vector(embedding: list, topk: int = None) -> Any:
    if topk is None:
        topk = DEFAULT_TOPK
    if not (COG_SEARCH_ENDPOINT and COG_SEARCH_KEY and COG_SEARCH_INDEX):
        raise RuntimeError("COG_SEARCH_ENDPOINT or COG_SEARCH_KEY or COG_SEARCH_INDEX not set")
    api_ver = COG_SEARCH_API_VERSION
    url = f"{COG_SEARCH_ENDPOINT}/indexes/{COG_SEARCH_INDEX}/docs/search?api-version={api_ver}"
    headers = {"api-key": COG_SEARCH_KEY, "Content-Type": "application/json"}
    vector_field = os.environ.get("SEARCH_VECTOR_FIELD") or os.environ.get("COG_SEARCH_VECTOR_FIELD") or COG_SEARCH_VECTOR_FIELD

    payload = {
        # vectorQueries (2024-07-01+)
        "count": True,
        "select": os.environ.get("SEARCH_SELECT_FIELDS") or SEARCH_SELECT_FIELDS,
        "vectorQueries": [
            {
                "kind": "vector",
                "vector": embedding,
                "fields": vector_field,
                "k": topk
            }
        ]
    }
    return request_json("post", url, headers=headers, json_payload=payload, timeout=120)


def text_search(query: str, topk: int = None) -> Any:
    if topk is None:
        topk = DEFAULT_TOPK
    if not (COG_SEARCH_ENDPOINT and COG_SEARCH_KEY and COG_SEARCH_INDEX):
        raise RuntimeError("COG_SEARCH_ENDPOINT or COG_SEARCH_KEY or COG_SEARCH_INDEX not set")
    api_ver = COG_SEARCH_API_VERSION
    url = f"{COG_SEARCH_ENDPOINT}/indexes/{COG_SEARCH_INDEX}/docs/search?api-version={api_ver}"
    headers = {"api-key": COG_SEARCH_KEY, "Content-Type": "application/json"}
    payload = {
        "search": query or "*",
        "top": topk
    }
    return request_json("post", url, headers=headers, json_payload=payload, timeout=REQUEST_TIMEOUT)


def parse_search_hits(search_json: Any) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    if not search_json:
        return hits
    if isinstance(search_json, dict) and "value" in search_json and isinstance(search_json["value"], list):
        for item in search_json["value"]:
            entry = {
                "id": item.get("id") or item.get("docId") or item.get("metadata_storage_path") or item.get("@search.documentkey"),
                "score": item.get("@search.score") or item.get("score") or None,
                "text": item.get("text") or item.get("content") or item.get("description") or item.get("normalized_text") or item.get("ocr_text") or None,
                "raw": item
            }
            hits.append(entry)
        return hits
    if isinstance(search_json, dict) and "results" in search_json:
        for r in search_json["results"]:
            hits.append({"id": r.get("id"), "score": r.get("score"), "text": r.get("text") or None, "raw": r})
        return hits
    if isinstance(search_json, list):
        for item in search_json:
            hits.append({"id": item.get("id") if isinstance(item, dict) else None, "score": item.get("score") if isinstance(item, dict) else None, "text": item.get("text") if isinstance(item, dict) else to_str_safe(item), "raw": item})
        return hits
    hits.append({"id": None, "score": None, "text": to_str_safe(search_json), "raw": search_json})
    return hits


def dedupe_results(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for h in hits:
        key = h.get("id") or to_str_safe(h.get("text", ""))[:200]
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
    return out


def get_doc_by_id(doc_id: str) -> Dict[str, Any]:
    if not doc_id:
        return {"id": None, "text": None}
    if COG_SEARCH_ENDPOINT and COG_SEARCH_KEY:
        url = f"{COG_SEARCH_ENDPOINT}/indexes/{COG_SEARCH_INDEX}/docs('{doc_id}')?api-version=2021-04-30-Preview"
        headers = {"api-key": COG_SEARCH_KEY}
        try:
            resp = request_json("get", url, headers=headers)
            if isinstance(resp, dict):
                return {"id": resp.get("id") or doc_id, "text": resp.get("content") or resp.get("ocr_text") or resp.get("text") or resp.get("description") or None, "raw": resp}
        except Exception:
            logger.debug("get_doc_by_id: fallback to minimal record")
    return {"id": doc_id, "text": None}


# ---------------------------
# Prompt builder for reranking / generation
# ---------------------------
def build_rerank_prompt(query: str, docs: List[Dict[str, Any]]) -> Tuple[str, str]:
    system_prompt = (
        "Você é um assistente que responde em Português. Use apenas as fontes fornecidas para construir a resposta. "
        "Retorne EXCLUSIVAMENTE um JSON válido com a estrutura: {\"text\": \"resumo curto\", \"bullets\": [\"acao1\", \"acao2\"]}. "
        "Se não houver informação suficiente, retorne {\"text\": \"... (explique o que falta)\", \"bullets\": []}. "
        "Não inclua campos extras fora desse JSON top-level."
    )
    doc_texts = []
    for i, d in enumerate(docs, start=1):
        src = None
        if isinstance(d.get("raw"), dict):
            src = d["raw"].get("metadata_storage_path") or d["raw"].get("source_file") or d["raw"].get("source")
        src = src or d.get("id")
        text = d.get("text") or ""
        one = f"Fonte {i} | id: {src or d.get('id')}\n{text[:2000]}"
        doc_texts.append(one)
    user_prompt = (
        f"Pergunta: {query}\n\n"
        "Fontes:\n" + ("\n\n---\n\n".join(doc_texts) if doc_texts else "Nenhuma fonte disponível.\n") + "\n\n"
        "Tarefa: Responda em Português COMPACTAMENTE e em formato JSON conforme a estrutura pedida. Cite fontes entre colchetes se for usar trechos (ex: [1])."
    )
    return system_prompt, user_prompt


# ---------------------------
# Central processing flow
# ---------------------------
def process_query(query: str, topk: int = DEFAULT_TOPK, debug: bool = False) -> Dict[str, Any]:
    result: Dict[str, Any] = {"query": query, "topK": topk, "used_embedding": False, "fallback_to_text": False, "hits_count": 0, "hits": [], "answer": None, "error": None}
    if not query or not str(query).strip():
        return {"error": "missing 'query' or query is empty", "status": 400}

    embedding = None
    try:
        embedding = get_embedding(query)
        result["used_embedding"] = True
    except Exception as e:
        logger.warning("Embedding failed, will fallback to text search: %s", e)
        result["embedding_error"] = to_str_safe(e)
        result["used_embedding"] = False

    try:
        if embedding and not FORCE_TEXT_SEARCH:
            try:
                search_json = search_vector(embedding, topk=topk)
            except Exception as e:
                logger.warning("Vector search failed, fallback to text search: %s", e)
                result["fallback_to_text"] = True
                search_json = text_search(query, topk=topk)
        else:
            result["fallback_to_text"] = True
            search_json = text_search(query, topk=topk)

        hits = parse_search_hits(search_json)
        hits = dedupe_results(hits)
        result["hits_count"] = len(hits)
        result["hits"] = hits
        if debug:
            result["search_raw"] = search_json
    except Exception as e:
        logger.exception("Search failed")
        return {"error": f"search failed: {e}", "status": 500}

    # fetch docs best-effort
    full_docs = []
    for h in result["hits"][:min(MAX_TOPK, topk)]:
        doc_id = h.get("id")
        if doc_id:
            try:
                doc_full = get_doc_by_id(doc_id)
                merged = {**h, **(doc_full or {})}
                full_docs.append(merged)
            except Exception as e:
                logger.debug("Failed fetch doc %s: %s", doc_id, e)
                h["_fetch_error"] = to_str_safe(e)
                full_docs.append(h)
        else:
            full_docs.append(h)

    # AOAI generation: prefer structured JSON per prompt
    try:
        system_prompt, user_prompt = build_rerank_prompt(query, full_docs[:5])
        chat_resp = call_aoai_chat(system_prompt, user_prompt)
        raw_content = extract_chat_content(chat_resp)
        logger.debug("AOAI raw content: %s", to_str_safe(raw_content, 2000))
        parsed = parse_json_from_text(raw_content)
        # normalize parsed result
        if isinstance(parsed, dict):
            # ensure keys
            if "text" not in parsed and "answer" in parsed:
                parsed = {"text": parsed.get("answer"), **{k: v for k, v in parsed.items() if k != "answer"}}
            result["answer"] = parsed
        elif isinstance(parsed, list):
            result["answer"] = {"text": json.dumps(parsed, ensure_ascii=False), "bullets": []}
        else:
            # numbers or strings -> wrap
            result["answer"] = {"text": to_str_safe(parsed), "bullets": []}
    except Exception as e:
        logger.warning("AOAI generation failed: %s", e)
        # fallback: try quick synthesizer or build minimal summary
        try:
            synth = synthesize_answer(query, full_docs[:5])
            # if synth returned something likely textual, wrap into JSON
            result["answer"] = {"text": synth, "bullets": []}
            result["generation_error"] = to_str_safe(e)
        except Exception as e2:
            # ultimate fallback: concatenate first paragraphs
            summary_lines = []
            for idx, d in enumerate(full_docs[:5], start=1):
                t = d.get("text") or ""
                one = (t.replace("\n", " ")[:400] + "...") if t else ""
                summary_lines.append(f"{idx}. {one} (id={d.get('id')})")
            fallback = "Resumo fallback basado nos documentos encontrados:\n\n" + "\n\n".join(summary_lines)
            result["answer"] = {"text": fallback}
            result["generation_error"] = to_str_safe(e)

    if debug:
        result["debug"] = {"used_embedding": result.get("used_embedding"), "fallback_to_text": result.get("fallback_to_text"), "hits_count": result.get("hits_count")}
    return result


# ---------------------------
# Public API for Azure Function wrapper (compact output)
# ---------------------------
def _make_compact_response(full: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extrai somente os campos que o usuário pediu.
    """
    return {
        "query": full.get("query"),
        "topK": full.get("topK"),
        "used_embedding": bool(full.get("used_embedding")),
        "fallback_to_text": bool(full.get("fallback_to_text")),
        "hits_count": int(full.get("hits_count", 0)),
        "answer": full.get("answer")
    }

def generate_client_response(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Entrada: body JSON do POST.
    - body["query"] - string
    - body["topK"] (opcional)
    - body["debug"] (opcional)
    - body["compact"] (opcional, default True) -> se False, retorna o objeto completo
    """
    query = ""
    try:
        query = (body.get("query") or body.get("q") or "").strip()
    except Exception:
        query = ""
    try:
        topk = int(body.get("topK", body.get("topk", DEFAULT_TOPK)))
    except Exception:
        topk = DEFAULT_TOPK
    topk = max(1, min(MAX_TOPK, topk))
    debug = bool(body.get("debug", False))
    compact = body.get("compact", True)  # default: True -> retorna só o resumo
    try:
        full_resp = process_query(query, topk=topk, debug=debug)
    except Exception as e:
        logger.exception("generate_client_response failed")
        return {"error": to_str_safe(e), "status": 500}

    # se pedirem explicitamente o objeto completo, devolve tudo (útil para debug)
    if compact in (False, "false", "False", 0, "0"):
        return full_resp

    # caso contrário devolve só os campos resumidos
    compacted = _make_compact_response(full_resp)
    return compacted


def handle_search_request(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    função wrapper usada pela Azure Function (ex: __init__.py chama handle_search_request(body))
    """
    return generate_client_response(body)


if __name__ == "__main__":
    test_q = "Qual é a previsão para o projeto de obras na avenida X?"
    print("Running quick local test (no network checks).")
    try:
        out = process_query(test_q, topk=3, debug=True)
        print(json.dumps(out, indent=2, ensure_ascii=False))
    except Exception as e:
        logger.exception("Local test failed: %s", e)
        print({"error": str(e)})
