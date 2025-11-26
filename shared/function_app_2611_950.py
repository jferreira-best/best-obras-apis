# function_app.py — versão completa com _find_explicit_policy_statements definido

import os
import json
import time
import logging
import re
import unicodedata
import copy
import shelve
from typing import Any, Dict, List, Optional, Union, Tuple
from collections import Counter
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse, unquote
import requests
import azure.functions as func

# Inicializa o App do Azure Functions
app = func.FunctionApp()



# =========================
# LOG
# =========================
logger = logging.getLogger("function_app_debug")
logger.setLevel(logging.INFO)

def _filename_from_source(src: str) -> str:
    if not src:
        return ""
    if src.startswith(("http://", "https://")):
        path = urlparse(src).path
        name = os.path.basename(path)
        return unquote(name) or src
    norm = src.replace("\\", "/")
    return norm.rsplit("/", 1)[-1]

def _safe_int_env(key: str, default: int) -> int:
    val = os.environ.get(key)
    if val is None or val == "":
        return default
    try:
        return int(val)
    except ValueError:
        logging.error("ERROR: Environment variable %s has invalid value '%s'. Using default %s.", key, val, default)
        return default

def _safe_str_env(key: str, default: str) -> str:
    val = os.environ.get(key)
    if val is None:
        return default
    return val

def _mask_secret(s: str) -> str:
    if not s:
        return "(empty)"
    s = str(s)
    if len(s) <= 8:
        return s[0:1] + "*****" + s[-1:]
    return s[:6] + "..." + s[-4:]

def resolve_env_map() -> Dict[str, Any]:
    env = os.environ
    SEARCH_ENDPOINT     = env.get("SEARCH_ENDPOINT") or env.get("COG_SEARCH_ENDPOINT")
    SEARCH_API_KEY      = env.get("SEARCH_API_KEY") or env.get("COG_SEARCH_KEY")
    SEARCH_INDEX        = env.get("SEARCH_INDEX") or env.get("COG_SEARCH_INDEX") or env.get("SEARCH_INDEX")
    OPENAI_API_BASE     = env.get("OPENAI_API_BASE") or env.get("AOAI_ENDPOINT") or env.get("AZURE_OPENAI_ENDPOINT")
    OPENAI_API_KEY      = env.get("OPENAI_API_KEY") or env.get("AOAI_API_KEY") or env.get("AZURE_OPENAI_KEY")
    OPENAI_DEPLOYMENT   = env.get("OPENAI_DEPLOYMENT") or env.get("AOAI_CHAT_DEPLOYMENT") or env.get("OPENAI_CHAT_DEPLOYMENT")
    FUNCTIONS_WORKER_RUNTIME = env.get("FUNCTIONS_WORKER_RUNTIME")
    AzureWebJobsStorage = env.get("AzureWebJobsStorage") or env.get("WEBSITE_CONTENTAZUREFILECONNECTIONSTRING")
    return {
        "SEARCH_ENDPOINT": SEARCH_ENDPOINT,
        "SEARCH_API_KEY": SEARCH_API_KEY,
        "SEARCH_INDEX": SEARCH_INDEX,
        "OPENAI_API_BASE": OPENAI_API_BASE,
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "OPENAI_DEPLOYMENT": OPENAI_DEPLOYMENT,
        "FUNCTIONS_WORKER_RUNTIME": FUNCTIONS_WORKER_RUNTIME,
        "AzureWebJobsStorage": AzureWebJobsStorage,
    }

def log_config(debug_flag: bool=False) -> None:
    cfg = resolve_env_map()
    logger.info("===== CONFIG RESOLVED (masked) =====")
    for k, v in cfg.items():
        logger.info("%s = %s", k, _mask_secret(v))
    logger.info("===== END CONFIG =====")
    if debug_flag:
        logger.info("Note: code checks both SEARCH_* and COG_* names, and OPENAI_* and AOAI_* groups.")

# --- Config / environment ---
LOG_LEVEL = _safe_str_env("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)

COG_SEARCH_ENDPOINT    = _safe_str_env("COG_SEARCH_ENDPOINT", "").rstrip("/")
COG_SEARCH_KEY         = _safe_str_env("COG_SEARCH_KEY", "")
COG_SEARCH_INDEX       = _safe_str_env("COG_SEARCH_INDEX", "")
COG_SEARCH_API_VERSION = _safe_str_env("COG_SEARCH_API_VERSION", "2024-07-01")

DEFAULT_TOPK           = _safe_int_env("DEFAULT_TOPK", 6)

ENABLE_SEMANTIC        = _safe_str_env("ENABLE_SEMANTIC", "true").lower() in ("1","true","yes","on")
COG_SEARCH_SEM_CONFIG  = _safe_str_env("COG_SEARCH_SEM_CONFIG", "")
SEARCH_FIELDS          = _safe_str_env("SEARCH_FIELDS", "doc_title,text")

AOAI_ENDPOINT          = _safe_str_env("AOAI_ENDPOINT", "").rstrip("/")
AOAI_API_KEY           = _safe_str_env("AOAI_API_KEY", "")
AOAI_EMB_DEPLOYMENT    = _safe_str_env("AOAI_EMB_DEPLOYMENT", "")
AOAI_CHAT_DEPLOYMENT   = _safe_str_env("AOAI_CHAT_DEPLOYMENT", "")
AOAI_API_VERSION       = _safe_str_env("AOAI_API_VERSION", "2023-10-01")

EMBED_DIM              = _safe_int_env("EMBED_DIM", 3072)
OPENAI_API_KEY         = _safe_str_env("OPENAI_API_KEY", "")
OPENAI_MODEL           = _safe_str_env("OPENAI_MODEL", "gpt-4o-mini")

HTTP_TIMEOUT_SHORT     = _safe_int_env("HTTP_TIMEOUT_SHORT", 8)
HTTP_TIMEOUT_LONG      = _safe_int_env("HTTP_TIMEOUT_LONG", 20)

EMB_CACHE_FILE         = _safe_str_env("EMB_CACHE_FILE", "/tmp/emb_cache.db")

# ---------- GUARDRAILS NOVOS (menos restritivos) ----------
#RELEVANCE_THRESHOLD_HITS = float(os.getenv("RELEVANCE_THRESHOLD_HITS", "0.18"))  # era 0.28
RELEVANCE_THRESHOLD_HITS = float(os.getenv("RELEVANCE_THRESHOLD_HITS", "0.10"))
MIN_QUOTES_REQUIRED      = int(os.getenv("MIN_QUOTES_REQUIRED", "2"))            # era 3
ALLOW_COMPLETION_WHEN_WEAK = os.getenv("ALLOW_COMPLETION_WHEN_WEAK", "true").lower() in ("1","true","yes","on")

def _strip_accents(s: str) -> str:
    if not s: 
        return ""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")

# Palavras do domínio (sem acentos + termos de água/solar)
_DOMAIN_KEYWORDS_RAW = [
    "obra","obras","manutencao","seguranca","engenharia","projeto",
    "canteiro","nr","manual","procedimento","instalacao","elevador","incendio",
    "aquecimento","chuva","demanda","inspecao","checklist","construcao","servicos",
    # novos para seus casos:
    "agua","potavel","nao potavel","qualidade","reservatorio","termico","coletores","solar","hidraulica"
]

DOMAIN_KEYWORDS = [k.lower() for k in _DOMAIN_KEYWORDS_RAW]
# --------------------------------------

STOPWORDS = {
    "o","a","os","as","de","do","da","dos","das","que",
    "é","e","ou","um","uma","para","por","em","no","na",
    "nos","nas","com","sem","se","ao","à","às","aos","não",
    "qual","quais","como","onde","quando"
}

# --- HTTP helper ---
def _http_post(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: int = HTTP_TIMEOUT_LONG) -> Dict[str, Any]:
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text}")
    return r.json()

# --- Text helpers ---
def _clean_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\r", " ")
    s = re.sub(r"\n+", " ", s)
    s = re.sub(r"\s{2,}", " ", s)
    s = re.sub(r"(\w)-\s+(\w)", r"\1\2", s)
    return s.strip()

def _tokenize_pt(s: str) -> List[str]:
    s = (s or "").lower()
    s = re.sub(r"[^\wà-úÀ-ÚçÇ-]+", " ", s)
    toks = [t for t in s.split() if len(t) > 1]
    return toks

# ---------------------------------------------------------------------
# Embeddings + cache
# ---------------------------------------------------------------------
def _embedding_or_none(text: str) -> Optional[List[float]]:
    if not text:
        return None
    try:
        if AOAI_ENDPOINT and AOAI_API_KEY and AOAI_EMB_DEPLOYMENT:
            #url = f"{AOAI_ENDPOINT}/openai/deployments/{AOAI_EMB_DEPLOYMENT}/embeddings?api-version=2023-05-15"
            url = f"{AOAI_ENDPOINT}/openai/deployments/{AOAI_EMB_DEPLOYMENT}/embeddings?api-version={AOAI_API_VERSION}"
            headers = {"api-key": AOAI_API_KEY, "Content-Type": "application/json"}
            payload = {"input": text}
            r = requests.post(url, headers=headers, json=payload, timeout=HTTP_TIMEOUT_SHORT)
            r.raise_for_status()
            vec = r.json()["data"][0]["embedding"]
            if len(vec) > EMBED_DIM:
                vec = vec[:EMBED_DIM]
            elif len(vec) < EMBED_DIM:
                vec += [0.0] * (EMBED_DIM - len(vec))
            return vec
    except Exception as e:
        logger.warning("embedding error (AOAI): %s", e)
    try:
        if OPENAI_API_KEY:
            url = "https://api.openai.com/v1/embeddings"
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
            #payload = {"model": "text-embedding-3-small", "input": text}
            payload = {"model": "text-embedding-3-large", "input": text}
            r = requests.post(url, headers=headers, json=payload, timeout=HTTP_TIMEOUT_SHORT)
            r.raise_for_status()
            data = r.json()
            return data.get("data", [{}])[0].get("embedding")
    except Exception as e:
        logger.warning("embedding error (OpenAI): %s", e)
    return None

def _persisted_embedding(q: str) -> Optional[Tuple[float, ...]]:
    return None

def _store_persisted_embedding(q: str, tup: Tuple[float, ...]):
    pass

@lru_cache(maxsize=1024)
def _cached_query_embedding_tuple(q: str) -> Optional[Tuple[float, ...]]:
    p = _persisted_embedding(q)
    if p:
        return p
    vec = _embedding_or_none(q)
    tup = tuple(vec) if vec else None
    if tup:
        _store_persisted_embedding(q, tup)
    return tup

def _get_query_embedding(query: str) -> Optional[List[float]]:
    tup = _cached_query_embedding_tuple(query)
    return list(tup) if tup else None

# ---------------------------------------------------------------------
# Chat/LLM wrapper
# ---------------------------------------------------------------------
def _call_api_with_messages(messages_to_send: List[Dict[str, str]], max_tokens: int = 400) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    try:
        if AOAI_ENDPOINT and AOAI_API_KEY and AOAI_CHAT_DEPLOYMENT:
            url = f"{AOAI_ENDPOINT}/openai/deployments/{AOAI_CHAT_DEPLOYMENT}/chat/completions?api-version={AOAI_API_VERSION}"
            headers = {"api-key": AOAI_API_KEY, "Content-Type": "application/json"}
            payload = {"messages": messages_to_send, "max_tokens": max_tokens, "temperature": 0.0}
            r = requests.post(url, headers=headers, json=payload, timeout=HTTP_TIMEOUT_LONG)
            r.raise_for_status()
            resp = r.json()
            try:
                txt = resp["choices"][0]["message"].get("content")
            except Exception:
                txt = resp.get("choices", [{}])[0].get("text")
            fr = resp["choices"][0].get("finish_reason") if resp.get("choices") else None
            return resp, fr, txt
        if OPENAI_API_KEY:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
            payload = {"model": OPENAI_MODEL, "messages": messages_to_send, "max_tokens": max_tokens, "temperature": 0.0}
            r = requests.post(url, headers=headers, json=payload, timeout=HTTP_TIMEOUT_LONG)
            r.raise_for_status()
            resp = r.json()
            txt = resp["choices"][0]["message"].get("content")
            fr = resp["choices"][0].get("finish_reason")
            return resp, fr, txt
    except Exception as e:
        logger.exception("Error calling LLM: %s", e)
    return None, None, None

# ---------------------------------------------------------------------
# Truncation helpers (mantidos)
# ---------------------------------------------------------------------
def _looks_truncated(text: str) -> bool:
    if not text or not text.strip():
        return True
    t = text.rstrip()
    if re.search(r'[\.!?]["\')\]]?\s*$', t):
        return False
    if t.endswith("..."):
        return True
    return True

def _complete_truncated(content: str, messages: List[Dict[str, str]], call_api_fn, max_retries: int = 2) -> str:
    if not _looks_truncated(content):
        return content
    for attempt in range(max_retries):
        finish_instruction = {
            "role": "user",
            "content": (
                "A seguir está uma resposta parcial gerada. COMPLETE APENAS a continuação, "
                "finalize a frase. Não repita o que já foi dito; escreva somente a continuação necessária em 1–2 frases."
            )
        }
        follow_messages = copy.deepcopy(messages)
        follow_messages.append({"role": "assistant", "content": content})
        follow_messages.append(finish_instruction)
        resp, finish_reason, cont_text = call_api_fn(follow_messages, max_tokens=200)
        cont_text = (cont_text or "").strip()
        if cont_text:
            if content.endswith(" ") or content.endswith("\n"):
                new_content = content + cont_text
            else:
                new_content = content.rstrip() + " " + cont_text
            if not _looks_truncated(new_content):
                return new_content
            content = new_content
        else:
            break
    return content

# ---------------------------------------------------------------------
# SUMARIZAÇÃO 100% EXTRATIVA
# ---------------------------------------------------------------------
def _call_llm_summarize(question: str, quotes: List[Dict[str, str]], compact: bool = False) -> Optional[str]:
    """
    Gera uma resposta EXTRATIVA a partir dos trechos encontrados.
    Se não houver evidência suficiente nos trechos, o modelo deve responder apenas: NAO_ENCONTRADO
    """
    if not quotes:
        return None

    # Monta bloco de trechos a partir dos quotes reais
    trechos_list = []
    for i, q in enumerate(quotes, 1):
        txt = (q.get("text") or q.get("content") or "").strip()
        if not txt:
            continue
        if len(txt) > 1200:
            txt = txt[:1200].rstrip() + "..."
        src = q.get("source") or q.get("source_file") or q.get("doc_title") or ""
        trechos_list.append(f"[{i}] {txt}\nFonte: {os.path.basename(src)}")

    if not trechos_list:
        return None

    trechos_block = "\n\n".join(trechos_list)

    system = (
        "Você é um assistente de IA focado em responder perguntas usando documentos de referência.\n"
        "Siga TODAS as instruções abaixo com muito rigor:\n\n"
        "1) Baseie sua resposta **EXCLUSIVAMENTE** nas informações dos trechos fornecidos.\n"
        "2) É PROIBIDO **inventar** fatos ou usar conhecimento externo que não esteja nos trechos.\n"
        "3) **PERMITIDO FAZER CONEXÕES:** Você **DEVE** conectar o sentido da pergunta com o sentido dos trechos. Por exemplo, se a pergunta é 'quem autoriza' e o texto diz 'cabe ao CEPLAE deliberar', você deve entender que 'deliberar' é a resposta para 'autorizar'. O mesmo vale para 'fluxo de solicitação' e 'manifestação'.\n"
        "4) **REGRA DE NEGÓCIO (RISCO vs EMERGÊNCIA):** Se a pergunta for sobre um *risco* (ex: 'trinca', 'pode cair', 'parece que vai cair'), use a definição de 'Manutenção de Risco'. Se a pergunta for sobre um *evento que já ocorreu* (ex: 'caiu', 'desabou'), use a definição de 'Emergência' ou 'Queda de Muro'.\n"
        "5) Tente responder a pergunta da melhor forma possível usando os trechos. Se os trechos forem irrelevantes ou realmente não contiverem a resposta (mesmo após tentar fazer as conexões de sentido da regra #3), explique que não encontrou a informação específica nos documentos.\n"
        "6) Não repita a pergunta, não peça desculpas.\n"
        "7) Responda em português claro e objetivo.\n"
    )

    # --- CORREÇÃO 2: Removida a regra contraditória "NAO_ENCONTRADO" ---
    user = (
        f"Pergunta do usuário:\n"
        f"{question}\n\n"
        f"Trechos relevantes dos documentos (use APENAS essas informações para responder):\n"
        f"{trechos_block}\n\n"
        "Instruções para a resposta:\n"
        "- Se os trechos trazem a informação necessária, escreva uma resposta direta e concisa,\n"
        "  em 1 ou 2 parágrafos curtos, explicando o que a pergunta pede.\n"
        "- Não acrescente nada que não esteja claramente suportado pelos trechos.\n"
    )

    _, _, ans = _call_api_with_messages(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        max_tokens=220 if compact else 400
    )

    if ans:
        ans = ans.strip()
    return ans or None


# ---------------------------------------------------------------------
# Resumos auxiliares (mantidos)
# ---------------------------------------------------------------------
def _mini_summary_from_quotes(query: str, quotes: List[Dict[str, str]]) -> Optional[str]:
    if not quotes:
        return None
    qwords = [w.lower() for w in re.findall(r'\w+', query) if len(w) > 3]
    for q in quotes:
        text = _clean_text(q.get("text", ""))
        low = text.lower()
        if any(w in low for w in qwords[:4]):
            m = re.split(r'(?<=[\.\?\!])\s+', text)
            if m and m[0]:
                s = m[0].strip()
                return s[:220].rstrip() + ("..." if len(s) > 220 else "")
    first = _clean_text(quotes[0].get("text", ""))
    return first[:220].rstrip() + ("..." if len(first) > 220 else "")

def _build_bullets_from_quotes(quotes: List[Dict[str, str]], max_bullets: int = 3) -> List[str]:
    bullets = []
    for q in quotes[:max_bullets]:
        t = _clean_text(q.get("text", ""))
        b = (t[:120].rstrip() + ("..." if len(t) > 120 else ""))
        bullets.append(b)
    return bullets

# ---------------------------------------------------------------------
# Azure Cognitive Search
# ---------------------------------------------------------------------
def _vector_search(query: str, topk: int, search_index: str) -> List[Dict[str, Any]]:

    try:
        tup = _cached_query_embedding_tuple(query)
        vec = list(tup) if tup else None
    except Exception:
        vec = None
    if not vec:
        vec = _embedding_or_none(query)
        if not vec:
            return []
    #url = f"{COG_SEARCH_ENDPOINT}/indexes/{COG_SEARCH_INDEX}/docs/search?api-version={COG_SEARCH_API_VERSION}"
    url = f"{COG_SEARCH_ENDPOINT}/indexes/{search_index}/docs/search?api-version={COG_SEARCH_API_VERSION}"
    headers = {"api-key": COG_SEARCH_KEY, "Content-Type": "application/json"}
    payload = {
        "search": "*",
        "top": topk,
        "vectorQueries": [
            {"kind": "vector", "vector": vec, "k": topk, "fields": "content_vector"}
        ]
    }
    try:
        data = _http_post(url, headers, payload, timeout=HTTP_TIMEOUT_LONG)
        return data.get("value", [])
    except Exception as e:
        logger.warning("vector search http error: %s", e)
        return []

def _vector_search_with_vec(vec: List[float], topk: int) -> List[Dict[str, Any]]:
    if not vec:
        return []
    url = f"{COG_SEARCH_ENDPOINT}/indexes/{COG_SEARCH_INDEX}/docs/search?api-version={COG_SEARCH_API_VERSION}"
    headers = {"api-key": COG_SEARCH_KEY, "Content-Type": "application/json"}
    payload = {
        "search": "*",
        "top": topk,
        "vectorQueries": [
            {"kind": "vector", "vector": vec, "k": topk, "fields": "content_vector"}
        ]
    }
    try:
        data = _http_post(url, headers, payload, timeout=HTTP_TIMEOUT_LONG)
        return data.get("value", [])
    except Exception as e:
        logger.warning("vector_with_vec http error: %s", e)
        return []

def _text_search(query: str, topk: int, semantic_config: str,search_index: str, force_semantic: bool = False, return_raw: bool = False) -> Any:
    if not COG_SEARCH_ENDPOINT or not COG_SEARCH_KEY:
        return [] if not return_raw else {}
    #url = f"{COG_SEARCH_ENDPOINT}/indexes/{COG_SEARCH_INDEX}/docs/search?api-version={COG_SEARCH_API_VERSION}"
    url = f"{COG_SEARCH_ENDPOINT}/indexes/{search_index}/docs/search?api-version={COG_SEARCH_API_VERSION}"
    headers = {"api-key": COG_SEARCH_KEY, "Content-Type": "application/json"}
    base_payload: Dict[str, Any] = {"search": query, "top": topk, "searchFields": SEARCH_FIELDS}

    #semantic_on = ENABLE_SEMANTIC and (force_semantic or bool(COG_SEARCH_SEM_CONFIG))
    semantic_on = ENABLE_SEMANTIC and (force_semantic or bool(semantic_config))
    if semantic_on:
        base_payload.update({
            "queryType": "semantic",
            #"queryLanguage": "pt-BR",
            "answers": "extractive",
            "answersCount": 1,
            "captions": "extractive",
            "captionsHighlight": False
        })
        if semantic_config: # Usa a config vinda do parâmetro
            base_payload["semanticConfiguration"] = semantic_config
    else:
        base_payload["queryType"] = "simple"

    try:
        data = _http_post(url, headers, base_payload, timeout=HTTP_TIMEOUT_LONG)
    except Exception as e:
        msg = str(e)
        if "queryLanguage" in msg or "not a valid parameter for the operation 'search'" in msg:
            logger.info("semantic params rejected by service; retrying without semantic params")
            simple_payload = {k: v for k, v in base_payload.items()
                              if k not in ("queryType", "queryLanguage", "answers",
                                           "answersCount", "captions",
                                           "captionsHighlight", "semanticConfiguration")}
            simple_payload.update({"queryType": "simple"})
            data = _http_post(url, headers, simple_payload, timeout=HTTP_TIMEOUT_LONG)
        else:
            raise
    return data if return_raw else (data.get("value", []) if data else [])

# ---------------------------------------------------------------------
# Normalização/extração (mantido)
# ---------------------------------------------------------------------
def _normalize_hit(h: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": h.get("id"),
        #"score": h.get("@search.score"),
        #"score": (h.get("@search.rerankerScore") / 4.0) if h.get("@search.rerankerScore") else h.get("@search.score"),
        "score": (h.get("@search.rerankerScore") / 4.0) if h.get("@search.rerankerScore") else (h.get("@search.vectorSearchScore") or h.get("@search.score")),
        "text": _clean_text(h.get("text") or ""),
        "source_file": h.get("source_file"),
        "id_original": h.get("id_original"),
    }

_SENT_SPLIT = re.compile(r'(?<=[\.\!\?\:])\s+')

def _split_sentences(txt: str) -> List[str]:
    txt = _clean_text(txt)
    return [s.strip() for s in _SENT_SPLIT.split(txt) if s.strip()]

def _make_query_keyset(query: str):
    q_tokens_all = _tokenize_pt(query)
    q_tokens = [t for t in q_tokens_all if t not in STOPWORDS]
    q_terms = Counter(q_tokens_all)
    qset = set(q_tokens)
    return q_terms, qset, q_tokens

def _score_sentence(sent: str, q_terms: Counter, qset: Optional[set] = None, q_tokens_ordered: Optional[List[str]] = None) -> float:
    if qset is None:
        qset = {t for t in q_terms.keys() if t not in STOPWORDS}
    sent_clean = _clean_text(sent).lower()
    toks = [t for t in _tokenize_pt(sent) if t not in STOPWORDS]
    if not toks:
        return 0.0
    matched = qset & set(toks)
    presence = len(matched) / max(1, len(qset))
    overlap = sum(q_terms.get(t, 0) for t in toks) / (1.0 + (len(toks) ** 0.5))
    bigram_boost = 0.0
    if not q_tokens_ordered:
        q_tokens_ordered = list(qset)
    max_ngrams_to_count = 2
    ngram_matches = 0
    for n in (2, 3):
        if len(q_tokens_ordered) < n:
            continue
        for i in range(len(q_tokens_ordered) - n + 1):
            gram = " ".join(q_tokens_ordered[i:i+n])
            if re.search(r'\b' + re.escape(gram.lower()) + r'\b', sent_clean, flags=re.I):
                ngram_matches += 1
                if ngram_matches >= max_ngrams_to_count:
                    break
        if ngram_matches >= max_ngrams_to_count:
            break
    bigram_boost = min(1.2, 0.45 * ngram_matches)
    score = (2.2 * presence) + (1.0 * overlap) + bigram_boost
    return float(score)

def _window(sentences: List[str], i: int, radius: int = 1) -> str:
    a = max(0, i - radius)
    b = min(len(sentences), i + radius + 1)
    return _clean_text(" ".join(sentences[a:b]))

def _prefer_definition(query: str, text: str) -> Optional[str]:
    q = _clean_text(query).lower()
    terms = [t for t in _tokenize_pt(q) if t not in {"o","a","de","do","da","que","é"}]
    if not terms:
        return None
    target = r"(?:{})(?:\s+|:)".format(r"\s+".join(map(re.escape, terms[:3])))
    patterns = [
        rf"(?i)\b{target}\s*[:\-–]\s*([^\.]{{20,240}})\.",
        rf"(?i)o que é\s+{target}\s*\??\s*([^\.]{{20,240}})\.",
        rf"(?i)definiç(?:ão|oes)\s*[:\-–]\s*([^\.]{{20,240}})\.",
    ]
    t = _clean_text(text)
    for pat in patterns:
        m = re.search(pat, t)
        if m:
            return _clean_text(m.group(0))
    return None

def _extract_quotes(hits: List[Dict[str, Any]], query: str, per_doc: int = 3, max_quotes: int = 8) -> List[Dict[str, str]]:
    q_terms, qset, q_tokens = _make_query_keyset(query)
    exact_phrase = None
    m = re.search(r"(?i)o que\s+é\s+(.+?)\??$", _clean_text(query))
    if m:
        exact_phrase = _clean_text(m.group(1))
        parts = exact_phrase.split()
        if len(parts) > 5:
            exact_phrase = " ".join(parts[:5])

    def _has_key_term(text: str) -> bool:
        toks = set(_tokenize_pt(text))
        return len(qset & toks) > 0

    quotes: List[Dict[str, str]] = []
    seen: set = set()

    for h in hits:
        txt = h.get("text") or ""
        maybe_def = _prefer_definition(query, txt)
        if maybe_def:
            snip = _clean_text(maybe_def)
            if 40 <= len(snip) <= 420 and snip not in seen and _has_key_term(snip):
                quotes.append({"source": h.get("source_file") or h.get("id_original") or "", "text": snip})
                seen.add(snip)
                if len(quotes) >= max_quotes:
                    return quotes

    if exact_phrase:
        for h in hits:
            txt = h.get("text") or ""
            sents = _split_sentences(txt)
            for i, s in enumerate(sents):
                snip = _clean_text(_window(sents, i, radius=1))
                if len(snip) < 20 or snip in seen:
                    continue
                if re.search(re.escape(exact_phrase), snip, flags=re.I):
                    quotes.append({"source": h.get("source_file") or h.get("id_original") or "", "text": snip})
                    seen.add(snip)
                    if len(quotes) >= max_quotes:
                        return quotes

    for h in hits:
        src = h.get("source_file") or h.get("id_original") or ""
        is_gloss = bool(re.search(r'gloss[aá]rio', src, flags=re.I))
        sents = _split_sentences(h.get("text") or "")
        if not sents:
            continue
        scored = []
        for i, s in enumerate(sents):
            sc = _score_sentence(s, q_terms, qset, q_tokens)
            if is_gloss:
                sc += 0.15
            scored.append((i, sc))
        scored.sort(key=lambda x: x[1], reverse=True)
        taken = 0
        for i, sc in scored:
            if sc <= 0:
                break
            snip = _window(sents, i, radius=1)
            snip = _clean_text(snip)
            if len(snip) < 20:
                continue
            if snip in seen:
                continue
            if not _has_key_term(snip):
                if not any(tok.upper() in snip.upper() for tok in q_tokens):
                    continue
            quotes.append({"source": src, "text": snip})
            seen.add(snip)
            taken += 1
            if taken >= per_doc or len(quotes) >= max_quotes:
                break
        if len(quotes) >= max_quotes:
            break
    return quotes

def _prioritize_sources(sources, limit=4):
    seen_keys = set()
    gloss, others = [], []
    for s in sources:
        if not s:
            continue
        base = os.path.basename(s).lower()
        if base in seen_keys:
            continue
        seen_keys.add(base)
        if re.search(r'gloss[aá]rio', s, flags=re.I):
            gloss.append(s)
        else:
            others.append(s)
    keep = gloss + others
    return keep[:limit]

def _validate_grounding(quotes: List[Dict[str, str]]) -> bool:
    return len(quotes) > 0

# -------- NOVA FUNÇÃO: acha sentenças “normativas” explícitas --------
_NORMATIVE_KEYWORDS = [
    "procedimento", "procedimentos", "norma", "normas", "política", "políticas",
    "deve", "devem", "obrigatório", "obrigatória", "obrigatórios",
    "responsável", "responsáveis", "responsabilidade",
    "é necessário", "é proibido", "não é permitido", "requisitos", "diretriz", "diretrizes"
]

def _find_explicit_policy_statements(nhits: List[Dict[str, Any]], query: str, q_tokens: List[str]) -> List[Dict[str, str]]:
    """
    Procura sentenças que aparentem ser instruções/mandatos/regras e que
    tenham interseção com tokens da própria query. Útil para perguntas de 'procedimento/norma'.
    """
    if not nhits:
        return []
    qset = {t.lower() for t in q_tokens if t and t.lower() not in STOPWORDS}
    out: List[Dict[str, str]] = []
    for h in nhits:
        raw = h.get("text") or ""
        if not raw:
            continue
        sents = _split_sentences(raw)
        for s in sents:
            ss = _clean_text(s)
            low = ss.lower()
            if any(k in low for k in _NORMATIVE_KEYWORDS):
                toks = {t for t in _tokenize_pt(low) if t not in STOPWORDS}
                if qset and (qset & toks):
                    out.append({
                        "source": h.get("source_file") or h.get("id_original") or h.get("id") or "",
                        "text": ss
                    })
    # dedup e limitação leve
    seen = set(); final=[]
    for q in out:
        t = q["text"]
        if t not in seen:
            seen.add(t)
            final.append(q)
        if len(final) >= 6:
            break
    return final
# ---------------------------------------------------------------------

def _build_bullets_from_quotes(quotes: List[Dict[str, str]], max_bullets: int = 3) -> List[str]:
    bullets: List[str] = []
    for q in quotes:
        t = _clean_text(q.get("text") or "")
        if len(t) > 220:
            t = t[:200].rstrip() + "..."
        bullets.append(t)
        if len(bullets) >= max_bullets:
            break
    return bullets

# ---------------------------------------------------------------------
# Guardrails helpers
# ---------------------------------------------------------------------
def _in_domain(query: str) -> bool:
    # normaliza acentos e remove pontuação
    q = _strip_accents((query or "").lower())
    q = re.sub(r"[^a-z0-9\s]+", " ", q)
    # aceita se houver qualquer keyword do domínio como substring inteira
    for kw in DOMAIN_KEYWORDS:
        # exige coincidência de palavra (evita falsos positivos em substrings longas)
        if re.search(rf"\b{re.escape(kw)}\b", q):
            return True
    return False

"""
def _filter_hits_by_relevance(nhits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for h in nhits:
        sc = h.get("score") or 0.0
        try:
            sc = float(sc)
        except Exception:
            sc = 0.0
        if sc >= RELEVANCE_THRESHOLD_HITS:
            out.append(h)
    return out
"""
def _filter_hits_by_relevance(nhits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not nhits:
        return []
    strong = []
    for h in nhits:
        sc = h.get("score") or 0.0
        try:
            sc = float(sc)
        except Exception:
            sc = 0.0
        if sc >= RELEVANCE_THRESHOLD_HITS:
            strong.append(h)

    # se nada passou no threshold, pega pelo menos os top 5
    if strong:
        return strong
    nhits_sorted = sorted(nhits, key=lambda x: (x.get("score") or 0.0), reverse=True)
    return nhits_sorted[:5]




# ---------------------------------------------------------------------
# Respostas finais
# ---------------------------------------------------------------------
NOT_FOUND_MSG   = "Não encontrei informações sobre esse tema nos documentos indexados."
OUT_OF_SCOPE_MSG= "Esse tema não faz parte do escopo dos documentos indexados (manuais/obras/segurança/manutencao)."

def _answer_not_found(compact: bool) -> Dict[str, Any]:
    txt = NOT_FOUND_MSG if compact else NOT_FOUND_MSG + " Tente reformular a pergunta com termos do domínio."
    return {"text": txt, "sources": []}

def _dedupe_sentences(text: str) -> str:
    import re as _re
    t = (text or "").strip()
    if not t:
        return t
    t = _re.sub(r'\s+', ' ', t)
    parts = _re.split(r'(?<=[\.\?!])\s+', t)
    seen, clean = set(), []
    for p in parts:
        k = p.strip().lower()
        if k and k not in seen:
            seen.add(k)
            clean.append(p.strip())
    return " ".join(clean)

def _render_sources(quotes: List[Dict[str, Any]], max_items: int = 4) -> List[str]:
    out = []
    for q in quotes[:max_items]:
        src = q.get("source") or q.get("source_file") or ""
        if src:
            out.append(src)
    return list(dict.fromkeys(out))

def _answer_from_quotes(query: str, quotes: List[Dict[str, Any]], compact: bool) -> Dict[str, Any]:
    # Se não há evidência e a query realmente parece fora do domínio, sinaliza fora do escopo
    if not quotes and not _in_domain(query):
        return {"text": OUT_OF_SCOPE_MSG, "sources": []}

    # Se há poucas evidências e está muito fraco, ainda podemos responder se ALLOW_COMPLETION_WHEN_WEAK=true
    if len(quotes) < MIN_QUOTES_REQUIRED and not ALLOW_COMPLETION_WHEN_WEAK:
        return _answer_not_found(compact)

    summary = _call_llm_summarize(query, quotes, compact=compact)
    if not summary or summary.strip().upper() == "NAO_ENCONTRADO":
        # Se não conseguiu montar resposta mesmo com evidências, cai para NOT FOUND (não alucina)
        return _answer_not_found(compact)

    summary = _dedupe_sentences(summary)
    return {"text": summary, "sources": _render_sources(quotes)}


# ---------------------------------------------------------------------
# Co-ocorrência, assembly e handler principal
# ---------------------------------------------------------------------
def _derive_query_key_tokens(query: str, min_token_len: int = 2) -> List[str]:
    toks = re.findall(r"[A-Za-zÀ-ú0-9]+", query or "")
    out = []
    for t in toks:
        tl = t.strip()
        if not tl or tl.lower() in STOPWORDS or len(tl) < min_token_len:
            continue
        out.append(_clean_text(tl))
    seen = set()
    ordered = []
    for t in out:
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered

def _collect_sentences_with_tokens(doc_text: str, tokens: List[str]) -> Dict[str, List[int]]:
    sentences = _split_sentences(doc_text)
    norm_sents = [_clean_text(s).lower() for s in sentences]
    token_positions = {t: [] for t in tokens}
    for i, ns in enumerate(norm_sents):
        for t in tokens:
            if t and (re.search(r'\b' + re.escape(t) + r'\b', ns)):
                token_positions[t].append(i)
    return token_positions

def _assemble_context_from_positions(sentences: List[str], positions: List[int], radius: int = 1) -> str:
    if not positions:
        return ""
    ranges = []
    for i in positions:
        a = max(0, i - radius)
        b = min(len(sentences), i + radius + 1)
        ranges.append((a, b))
    ranges.sort()
    merged = []
    cur_a, cur_b = ranges[0]
    for a, b in ranges[1:]:
        if a <= cur_b:
            cur_b = max(cur_b, b)
        else:
            merged.append((cur_a, cur_b))
            cur_a, cur_b = a, b
    merged.append((cur_a, cur_b))
    pieces = []
    for a, b in merged:
        pieces.append(" ".join(_clean_text(s) for s in sentences[a:b]))
    return " ".join(pieces).strip()

def _add_query_doc_cooccurrence(nhits: List[Dict[str, Any]], quotes: List[Dict[str,str]], query: str,
                                min_tokens_match: int = 1, radius:int = 1, max_added_docs:int = 4) -> List[Dict[str,str]]:
    q_tokens = _derive_query_key_tokens(query)
    if not q_tokens:
        return quotes
    added = []
    seen_texts = { _clean_text(q.get("text") or "") for q in quotes }
    for h in nhits:
        raw = h.get("text") or ""
        if not raw or len(raw.strip()) < 20:
            continue
        token_positions = _collect_sentences_with_tokens(raw, q_tokens)
        matched_tokens = [t for t, pos in token_positions.items() if pos]
        if len(matched_tokens) >= min_tokens_match:
            sents = _split_sentences(raw)
            all_positions = []
            for t in matched_tokens:
                all_positions.extend(token_positions[t])
            all_positions = sorted(set(all_positions))
            ctx = _assemble_context_from_positions(sents, all_positions, radius=radius)
            ctx_clean = _clean_text(ctx)
            if ctx_clean and len(ctx_clean) >= 30 and ctx_clean not in seen_texts:
                src = h.get("source_file") or h.get("id_original") or h.get("id") or ""
                added.append({"source": src, "text": ctx_clean})
                seen_texts.add(ctx_clean)
                if len(added) >= max_added_docs:
                    break
        raw_norm = re.sub(r'\s+', ' ', _clean_text(raw)).lower()
        token_spans = {}
        for t in q_tokens:
            t_norm = t.lower()
            token_spans[t] = []
            start = 0
            while True:
                idx = raw_norm.find(t_norm, start)
                if idx == -1:
                    break
                token_spans[t].append((idx, idx + len(t_norm)))
                start = idx + len(t_norm)
        spans_list = []
        for t, spans in token_spans.items():
            for sp in spans:
                spans_list.append((t, sp[0], sp[1]))
        spans_list.sort(key=lambda x: x[1])
        found_pair_ctx = None
        char_window_size = _safe_int_env("COOC_CHAR_WINDOW", 300)
        for i in range(len(spans_list)):
            t1, s1, e1 = spans_list[i]
            for j in range(i+1, min(i+6, len(spans_list))):
                t2, s2, e2 = spans_list[j]
                dist = s2 - e1 if s2 >= e1 else s1 - e2 if s1 >= e2 else 0
                if dist <= char_window_size:
                    try:
                        m1 = re.search(re.escape(t1), raw, flags=re.I)
                        m2 = re.search(re.escape(t2), raw, flags=re.I)
                        if m1 and m2:
                            a = min(m1.start(), m2.start())
                            b = max(m1.end(), m2.end())
                            a2 = max(0, a - 120)
                            b2 = min(len(raw), b + 120)
                            candidate = _clean_text(raw[a2:b2])
                            if candidate and len(candidate) >= 30 and candidate not in seen_texts:
                                found_pair_ctx = candidate
                    except Exception:
                        found_pair_ctx = None
                    break
            if found_pair_ctx:
                break
        if found_pair_ctx:
            src = h.get("source_file") or h.get("id_original") or h.get("id") or ""
            added.append({"source": src, "text": found_pair_ctx})
            seen_texts.add(found_pair_ctx)
            if len(added) >= max_added_docs:
                break
    if added:
        for a in added:
            logging.debug("CO-OCC ADD: %s -> %.160s", a.get("source"), a.get("text")[:160].replace("\n"," "))
        new_quotes = []
        for a in added:
            if a not in quotes:
                new_quotes.append(a)
        quotes = new_quotes + quotes
    return quotes

# ---------------------------------------------------------------------
# Handler principal
# ---------------------------------------------------------------------
def handle_search_request(body: Dict[str, Any]) -> Dict[str, Any]:
    import logging
    if body.get("debug"):
        log_config(debug_flag=True)

    query   = (body or {}).get("query") or ""
    topk    = int((body or {}).get("topK") or DEFAULT_TOPK)
    compact = bool((body or {}).get("compact", False))

    # --- NOVO AJUSTE DE REGRA DE NEGÓCIO (Forçar Emergência) ---
    query_lower = query.lower()
    if ("preso" in query_lower or "trancado" or "quebrou" in query_lower) and "elevador" in query_lower:
        logging.info("Regra de negócio: Forçando 'emergência' para consulta de elevador preso.")
        query = query + " emergência risco"
    
    # NOVAS LINHAS: Pega os valores do body (com um fallback seguro)
    # Note que não usamos mais COG_SEARCH_INDEX ou COG_SEARCH_SEM_CONFIG global
    req_index    = (body or {}).get("search_index") or COG_SEARCH_INDEX
    req_sem_cfg  = (body or {}).get("semantic_config") or COG_SEARCH_SEM_CONFIG

    if not req_index:
        raise ValueError("Parâmetro 'search_index' é obrigatório no JSON da requisição.")

    if _is_short_def_query(query):
        topk = max(topk, 12)

    used_embedding = False
    hits: List[Dict[str, Any]] = []
    sem_json = None

    def _harvest_semantic_evidence_local(sem_json_local: Any) -> List[Dict[str, str]]:
        quotes_local: List[Dict[str, str]] = []
        if not isinstance(sem_json_local, dict):
            return quotes_local
        ans = sem_json_local.get("@search.answers") or []
        for a in ans:
            txt = (a.get("text") or "").strip()
            if not txt:
                continue
            snip = _clean_text(txt)
            if 40 <= len(snip) <= 420:
                quotes_local.append({"source": "", "text": snip})
        vals = sem_json_local.get("value") or []
        for v in vals:
            caps = v.get("@search.captions") or []
            src = v.get("source_file") or v.get("id_original") or v.get("id") or ""
            for c in caps:
                txt = (c.get("text") or "").strip()
                if not txt:
                    continue
                snip = _clean_text(txt)
                if 40 <= len(snip) <= 420:
                    quotes_local.append({"source": src, "text": snip})
        seen = set()
        out: List[Dict[str, str]] = []
        for q in quotes_local:
            t = q["text"]
            if t not in seen:
                out.append(q)
                seen.add(t)
        return out[:5]

    with ThreadPoolExecutor(max_workers=2) as ex:
        future_vec  = ex.submit(_vector_search, query, topk, req_index)
        #future_text = ex.submit(_text_search, query, topk, req_sem_cfg, force_semantic=_is_short_def_query(query), return_raw=True, search_index=req_index)
        # Passa req_index como o 4º argumento posicional
        future_text = ex.submit(_text_search, query, topk, req_sem_cfg, req_index, force_semantic=_is_short_def_query(query), return_raw=True)
        try:
            sem_json = future_text.result(timeout=HTTP_TIMEOUT_LONG + 5)
            thits = (sem_json.get("value", []) if isinstance(sem_json, dict) else [])
        except Exception as e:
            logging.warning("text search failed or timed out: %s", e)
            sem_json = None
            thits = []

        try:
            vhits = future_vec.result(timeout=HTTP_TIMEOUT_SHORT + 10)
        except Exception as e:
            logging.warning("vector search failed or timed out: %s", e)
            vhits = []

        if vhits:
            hits.extend(vhits)
            used_embedding = True
        if thits:
            by_id = {}
            for h in (hits + thits):
                hid = h.get("id")
                if hid:
                    by_id[hid] = h
                else:
                    by_id[f"__noid__{id(h)}"] = h
            hits = list(by_id.values())

    nhits_all = [_normalize_hit(h) for h in hits]
    nhits = _filter_hits_by_relevance(nhits_all)

    sources = _prioritize_sources([h.get("source_file") or h.get("id_original") for h in nhits if h])

    def _find_explicit_policy_statements_wrapper():
        q_tokens_for_search = [t for t in re.findall(r"[A-Za-zÀ-ú0-9]+", query) if len(t) > 1]
        return _find_explicit_policy_statements(nhits, query, q_tokens_for_search)

    semantic_quotes = _harvest_semantic_evidence_local(sem_json) if isinstance(sem_json, dict) else []
    quotes_main = _extract_quotes(nhits, query, per_doc=3, max_quotes=8)

    seen_q = set()
    quotes: List[Dict[str, str]] = []
    for q in (semantic_quotes + quotes_main):
        snip = q.get("text") or ""
        if snip and snip not in seen_q:
            quotes.append(q)
            seen_q.add(snip)
    quotes = quotes[:8]

    explicit_quotes = _find_explicit_policy_statements_wrapper()
    if explicit_quotes:
        existing = { _clean_text(q.get("text") or "") for q in quotes }
        new_explicit = [q for q in explicit_quotes if _clean_text(q.get("text") or "") not in existing]
        if new_explicit:
            logging.debug("FOUND EXPLICIT POLICY QUOTES: %d", len(new_explicit))
            quotes = new_explicit + quotes

    quotes = _add_query_doc_cooccurrence(nhits, quotes, query, min_tokens_match=1, radius=1, max_added_docs=4)

    if quotes:
        q_terms_counter, qset_unique, q_tokens_ordered = _make_query_keyset(query)
        scored_quotes = []
        for q in quotes:
            txt = q.get("text") or ""
            sc = _score_sentence(txt, q_terms_counter, qset_unique, q_tokens_ordered)
            scored_quotes.append((sc, q))
        scored_quotes.sort(key=lambda x: x[0], reverse=True)
        quotes = [q for s, q in scored_quotes]

    if not quotes:
        try:
            sem_json2 = _text_search(query, max(topk, 15), force_semantic=True, return_raw=True)
            thits2 = (sem_json2.get("value", []) if isinstance(sem_json2, dict) else [])
            nhits2_all = [_normalize_hit(h) for h in thits2] if thits2 else []
            nhits2 = _filter_hits_by_relevance(nhits2_all)
            semantic_quotes2 = _harvest_semantic_evidence_local(sem_json2) if isinstance(sem_json2, dict) else []
            quotes2_main = _extract_quotes(nhits2, query, per_doc=3, max_quotes=8)
            seen2 = set()
            combined2: List[Dict[str, str]] = []
            for q in (semantic_quotes2 + quotes2_main):
                snip = q.get("text") or ""
                if snip and snip not in seen2:
                    combined2.append(q)
                    seen2.add(snip)
            quotes = combined2[:8]
            if nhits2 and not sources:
                sources = _prioritize_sources([h.get("source_file") or h.get("id_original") for h in nhits2 if h])
        except Exception as e:
            logging.warning("second text search failed: %s", e)

    result = _answer_not_found(compact) if not quotes else _answer_from_quotes(query, quotes, compact)

    return {
        "status": "ok",
        "versao":"v1.06",
        "query": query,
        "result": result
    }

# ---------------------------------------------------------------------
# Utilidade: detectar perguntas curtas de definição
# ---------------------------------------------------------------------
def _is_short_def_query(query: str) -> bool:
    q = _clean_text(query).lower()
    toks = _tokenize_pt(q)
    if len(toks) <= 6 and (
        q.startswith("o que é") or q.startswith("o que e") or
        q.startswith("defina") or q.startswith("conceito de")
    ):
        return True
    return False

# --- FUNÇÃO 1: A NOVA (Busca / RAG) ---
@app.function_name(name="search_obras")
@app.route(route="search_obras", auth_level=func.AuthLevel.ANONYMOUS)
def http_search_trigger(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Recebendo requisição de busca.')
    
    try:
        # 1. Parse do Body
        try:
            body = req.get_json()
        except ValueError:
            return func.HttpResponse(
                json.dumps({"error": "Invalid JSON body"}, ensure_ascii=False),
                mimetype="application/json",
                status_code=400
            )

        # 2. CHAMADA DA LÓGICA PRINCIPAL
        # Aqui conectamos o HTTP Trigger com o seu script de RAG
        result = handle_search_request(body)

        # 3. Retorno
        return func.HttpResponse(
            json.dumps(result, ensure_ascii=False),
            mimetype="application/json",
            status_code=200
        )

    except Exception as e:
        logging.exception(f"Erro crítico na function: {e}")
        return func.HttpResponse(
            json.dumps({"error": f"Erro interno: {str(e)}"}, ensure_ascii=False),
            mimetype="application/json",
            status_code=500
        )

# ---------------------------------------------------------------------
# Execução local
# ---------------------------------------------------------------------
if __name__ == "__main__":
    test_query = "O que é gestão de projetos segundo o PMBOK?"
    body = {"query": test_query, "topK": 10, "compact": True, "debug": True}
    out = handle_search_request(body)
    print(json.dumps(out, ensure_ascii=False, indent=2))
