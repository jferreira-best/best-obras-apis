import requests
import copy
import re
from typing import List, Dict, Any, Optional, Tuple
from functools import lru_cache
from shared import config
from shared.utils import clean_text, mask_secret

def _http_post(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: int = config.HTTP_TIMEOUT_LONG) -> Dict[str, Any]:
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if r.status_code >= 400:
        # Log para ajudar a debugar o payload caso dê erro novamente
        config.logger.error(f"Erro HTTP {r.status_code} na URL {url}. Payload: {payload}. Response: {r.text}")
        raise RuntimeError(f"HTTP {r.status_code}: {r.text}")
    return r.json()

# --- Embedding ---
def _embedding_or_none(text: str) -> Optional[List[float]]:
    if not text:
        return None
    try:
        if config.AOAI_ENDPOINT and config.AOAI_API_KEY and config.AOAI_EMB_DEPLOYMENT:
            url = f"{config.AOAI_ENDPOINT}/openai/deployments/{config.AOAI_EMB_DEPLOYMENT}/embeddings?api-version={config.AOAI_API_VERSION}"
            headers = {"api-key": config.AOAI_API_KEY, "Content-Type": "application/json"}
            payload = {"input": text}
            r = requests.post(url, headers=headers, json=payload, timeout=config.HTTP_TIMEOUT_SHORT)
            r.raise_for_status()
            vec = r.json()["data"][0]["embedding"]
            if len(vec) > config.EMBED_DIM:
                vec = vec[:config.EMBED_DIM]
            elif len(vec) < config.EMBED_DIM:
                vec += [0.0] * (config.EMBED_DIM - len(vec))
            return vec
    except Exception as e:
        config.logger.warning("embedding error (AOAI): %s", e)
    
    try:
        if config.OPENAI_API_KEY:
            url = "https://api.openai.com/v1/embeddings"
            headers = {"Authorization": f"Bearer {config.OPENAI_API_KEY}", "Content-Type": "application/json"}
            payload = {"model": "text-embedding-3-large", "input": text}
            r = requests.post(url, headers=headers, json=payload, timeout=config.HTTP_TIMEOUT_SHORT)
            r.raise_for_status()
            return r.json().get("data", [{}])[0].get("embedding")
    except Exception as e:
        config.logger.warning("embedding error (OpenAI): %s", e)
    return None

@lru_cache(maxsize=1024)
def get_cached_query_embedding_tuple(q: str) -> Optional[Tuple[float, ...]]:
    vec = _embedding_or_none(q)
    return tuple(vec) if vec else None

def get_query_embedding(query: str) -> Optional[List[float]]:
    tup = get_cached_query_embedding_tuple(query)
    return list(tup) if tup else None

# --- Chat LLM ---
def call_api_with_messages(messages_to_send: List[Dict[str, str]], max_tokens: int = 400) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    try:
        if config.AOAI_ENDPOINT and config.AOAI_API_KEY and config.AOAI_CHAT_DEPLOYMENT:
            url = f"{config.AOAI_ENDPOINT}/openai/deployments/{config.AOAI_CHAT_DEPLOYMENT}/chat/completions?api-version={config.AOAI_API_VERSION}"
            headers = {"api-key": config.AOAI_API_KEY, "Content-Type": "application/json"}
            payload = {"messages": messages_to_send, "max_tokens": max_tokens, "temperature": 0.0}
            r = requests.post(url, headers=headers, json=payload, timeout=config.HTTP_TIMEOUT_LONG)
            r.raise_for_status()
            resp = r.json()
            txt = resp["choices"][0]["message"].get("content")
            fr = resp["choices"][0].get("finish_reason")
            return resp, fr, txt
        
        if config.OPENAI_API_KEY:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {config.OPENAI_API_KEY}", "Content-Type": "application/json"}
            payload = {"model": config.OPENAI_MODEL, "messages": messages_to_send, "max_tokens": max_tokens, "temperature": 0.0}
            r = requests.post(url, headers=headers, json=payload, timeout=config.HTTP_TIMEOUT_LONG)
            r.raise_for_status()
            resp = r.json()
            txt = resp["choices"][0]["message"].get("content")
            fr = resp["choices"][0].get("finish_reason")
            return resp, fr, txt
    except Exception as e:
        config.logger.exception("Error calling LLM: %s", e)
    return None, None, None

# --- Azure Search ---
def vector_search(query: str, topk: int, search_index: str) -> List[Dict[str, Any]]:
    vec = get_query_embedding(query)
    if not vec:
        return []
    url = f"{config.COG_SEARCH_ENDPOINT}/indexes/{search_index}/docs/search?api-version={config.COG_SEARCH_API_VERSION}"
    headers = {"api-key": config.COG_SEARCH_KEY, "Content-Type": "application/json"}
    payload = {
        "search": "*", "top": topk,
        "vectorQueries": [{"kind": "vector", "vector": vec, "k": topk, "fields": "content_vector"}]
    }
    try:
        data = _http_post(url, headers, payload)
        return data.get("value", [])
    except Exception as e:
        config.logger.warning("vector search http error: %s", e)
        return []

def text_search(query: str, topk: int, semantic_config: str, search_index: str, force_semantic: bool = False, return_raw: bool = False) -> Any:
    if not config.COG_SEARCH_ENDPOINT or not config.COG_SEARCH_KEY:
        return [] if not return_raw else {}
    
    url = f"{config.COG_SEARCH_ENDPOINT}/indexes/{search_index}/docs/search?api-version={config.COG_SEARCH_API_VERSION}"
    headers = {"api-key": config.COG_SEARCH_KEY, "Content-Type": "application/json"}
    base_payload = {"search": query, "top": topk, "searchFields": config.SEARCH_FIELDS}

    semantic_on = config.ENABLE_SEMANTIC and (force_semantic or bool(semantic_config))
    
    if semantic_on:
        # CORREÇÃO AQUI: Em REST API, answers e captions são strings com pipes
        # Formato: "extractive|count-<numero>|threshold-<valor>"
        base_payload.update({
            "queryType": "semantic",
            "answers": "extractive|count-1",     # Corrigido
            "captions": "extractive|highlight-false" # Corrigido (removemos captionsHighlight)
        })
        if semantic_config:
            base_payload["semanticConfiguration"] = semantic_config
    else:
        base_payload["queryType"] = "simple"

    try:
        data = _http_post(url, headers, base_payload)
    except Exception as e:
        # Fallback: Se falhar com semantic, tenta simple
        if "queryType" in base_payload and base_payload["queryType"] == "semantic":
            config.logger.info("Semantic search failed, retrying with simple search.")
            simple = {k:v for k,v in base_payload.items() if k not in ("queryType", "semanticConfiguration", "answers", "captions")}
            simple["queryType"] = "simple"
            data = _http_post(url, headers, simple)
        else:
            raise
    return data if return_raw else (data.get("value", []) if data else [])