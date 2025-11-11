# function_app.py
# Versão consolidada: mantém toda a lógica original do seu script,
# aplica correções no regex, detecção/compleção de truncamento LLM,
# adiciona cache LRU para embeddings de query e paraleliza vector+text search.
#
# Substitua seu function_app.py por este arquivo e reinicie o serviço.

import os
import json
import time
import logging
import re
import unicodedata
import copy
from typing import Any, Dict, List, Optional, Union, Tuple
from collections import Counter
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

import requests

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)

COG_SEARCH_ENDPOINT    = os.environ.get("COG_SEARCH_ENDPOINT", "").rstrip("/")
COG_SEARCH_KEY         = os.environ.get("COG_SEARCH_KEY", "")
COG_SEARCH_INDEX       = os.environ.get("COG_SEARCH_INDEX", "")
COG_SEARCH_API_VERSION = os.environ.get("COG_SEARCH_API_VERSION", "2024-07-01")
DEFAULT_TOPK           = int(os.environ.get("DEFAULT_TOPK", "6"))

ENABLE_SEMANTIC       = os.environ.get("ENABLE_SEMANTIC", "true").lower() in ("1","true","yes","on")
COG_SEARCH_SEM_CONFIG = os.environ.get("COG_SEARCH_SEM_CONFIG", "")

SEARCH_FIELDS         = os.environ.get("SEARCH_FIELDS", "doc_title,text")

AOAI_ENDPOINT       = os.environ.get("AOAI_ENDPOINT", "").rstrip("/")
AOAI_API_KEY        = os.environ.get("AOAI_API_KEY", "")
AOAI_EMB_DEPLOYMENT = os.environ.get("AOAI_EMB_DEPLOYMENT", "")
AOAI_CHAT_DEPLOYMENT = os.environ.get("AOAI_CHAT_DEPLOYMENT", "")
AOAI_API_VERSION    = os.environ.get("AOAI_API_VERSION", "2023-10-01")
EMBED_DIM           = int(os.environ.get("EMBED_DIM", "3072"))

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

STRICT_RAG_ONLY     = True

STOPWORDS = {
    "o","a","os","as","de","do","da","dos","das","que",
    "é","e","ou","um","uma","para","por","em","no","na",
    "nos","nas","com","sem","se","ao","à","às","aos","não",
    "qual","quais","como","onde","quando"
}

# --- HTTP helper ---
def _http_post(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: int = 20) -> Dict[str, Any]:
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text}")
    return r.json()

# --- text cleaning/tokenizing ---
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

# --- embedding call (AOAI) ---
def _embedding_or_none(text: str) -> Optional[List[float]]:
    if not (AOAI_ENDPOINT and AOAI_API_KEY and AOAI_EMB_DEPLOYMENT):
        return None
    try:
        url = f"{AOAI_ENDPOINT}/openai/deployments/{AOAI_EMB_DEPLOYMENT}/embeddings?api-version=2023-05-15"
        headers = {"api-key": AOAI_API_KEY, "Content-Type": "application/json"}
        payload = {"input": text}
        r = requests.post(url, headers=headers, json=payload, timeout=25)
        r.raise_for_status()
        vec = r.json()["data"][0]["embedding"]
        if len(vec) > EMBED_DIM:
            vec = vec[:EMBED_DIM]
        elif len(vec) < EMBED_DIM:
            vec += [0.0] * (EMBED_DIM - len(vec))
        return vec
    except Exception as e:
        logging.warning(f"embedding error: {e}")
        return None

# --- LRU cache for query embeddings (string -> tuple) ---
@lru_cache(maxsize=1024)
def _cached_query_embedding_tuple(q: str) -> Optional[Tuple[float, ...]]:
    vec = _embedding_or_none(q)
    return tuple(vec) if vec else None

def _get_query_embedding(query: str) -> Optional[List[float]]:
    tup = _cached_query_embedding_tuple(query)
    return list(tup) if tup else None

def _is_short_def_query(query: str) -> bool:
    q = _clean_text(query).lower()
    toks = _tokenize_pt(q)
    if len(toks) <= 6 and (
        q.startswith("o que é") or q.startswith("o que e") or
        q.startswith("defina") or q.startswith("conceito de")
    ):
        return True
    return False

# ---------------------------------------------------------------------
# --- LLM call helpers, truncation detection and completion helpers ---
# ---------------------------------------------------------------------
def _looks_truncated(text: str) -> bool:
    """Detecta se o texto parece ter sido cortado (sem pontuação final adequada ou com reticências)."""
    if not text or not text.strip():
        return True
    t = text.rstrip()
    # termina com pontuação final que indica conclusão
    if re.search(r'[\.!?]["\')\]]?\s*$', t):
        return False
    # termina com reticências explícitas (tratado como truncado para forçar continuação opcional)
    if t.endswith("..."):
        return True
    # se não termina com pontuação, considerar truncado (conservador)
    return True

def _call_api_with_messages(messages_to_send: List[Dict[str, str]], max_tokens: int = 400) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    """
    Faz a chamada à API (Azure AOAI ou OpenAI) com a lista de messages e retorna:
      (full_response_json_or_None, finish_reason_or_None, extracted_text_or_None)
    """
    try:
        # Azure OpenAI (AOAI) - chat completions endpoint (deployment)
        if globals().get("AOAI_ENDPOINT") and globals().get("AOAI_API_KEY") and globals().get("AOAI_CHAT_DEPLOYMENT"):
            url = f"{AOAI_ENDPOINT}/openai/deployments/{AOAI_CHAT_DEPLOYMENT}/chat/completions?api-version={AOAI_API_VERSION}"
            headers = {"api-key": AOAI_API_KEY, "Content-Type": "application/json"}
            payload = {"messages": messages_to_send, "max_tokens": max_tokens, "temperature": 0.0}
            r = requests.post(url, headers=headers, json=payload, timeout=30)
            r.raise_for_status()
            resp = r.json()
            try:
                txt = resp["choices"][0]["message"].get("content")
            except Exception:
                txt = resp.get("choices", [{}])[0].get("text")
            fr = resp["choices"][0].get("finish_reason") if resp.get("choices") else None
            return resp, fr, txt

        # Azure-style OpenAI base (custom OPENAI_API_BASE with azure type)
        if os.environ.get("OPENAI_API_TYPE", "").lower() == "azure" and os.environ.get("OPENAI_API_BASE") and os.environ.get("OPENAI_API_KEY"):
            base_url = os.environ.get("OPENAI_API_BASE")
            headers = {"api-key": os.environ.get("OPENAI_API_KEY"), "Content-Type": "application/json"}
            payload = {"messages": messages_to_send, "max_tokens": max_tokens, "temperature": 0.0}
            r = requests.post(base_url, headers=headers, json=payload, timeout=30)
            r.raise_for_status()
            resp = r.json()
            try:
                txt = resp["choices"][0]["message"].get("content")
            except Exception:
                txt = resp.get("choices", [{}])[0].get("text")
            fr = resp["choices"][0].get("finish_reason") if resp.get("choices") else None
            return resp, fr, txt

        # Public OpenAI
        if globals().get("OPENAI_API_KEY"):
            url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
            payload = {
                "model": globals().get("OPENAI_MODEL") or "gpt-4o-mini",
                "messages": messages_to_send,
                "max_tokens": max_tokens,
                "temperature": 0.0,
            }
            r = requests.post(url, headers=headers, json=payload, timeout=30)
            r.raise_for_status()
            resp = r.json()
            txt = resp["choices"][0]["message"].get("content")
            fr = resp["choices"][0].get("finish_reason")
            return resp, fr, txt

    except Exception as e:
        logging.exception("Erro ao chamar API no helper _call_api_with_messages: %s", e)

    return None, None, None

def _complete_truncated(content: str, messages: List[Dict[str, str]], call_api_fn, max_retries: int = 2) -> str:
    """
    Se `content` parece truncado, pede ao modelo que complete a continuação.
    call_api_fn(messages) -> (data, finish_reason, extracted_text)
    """
    if not _looks_truncated(content):
        return content

    for attempt in range(max_retries):
        finish_instruction = {
            "role": "user",
            "content": (
                "A seguir está uma resposta parcial gerada. COMPLETE APENAS a continuação, "
                "finalize a frase e encerre a resposta com exatamente: \"Quer mais detalhes? (s/n)\". "
                "Não repita o que já foi dito; escreva somente a continuação necessária em 1–2 frases."
            )
        }

        follow_messages = copy.deepcopy(messages)
        # adiciona a resposta parcial como histórico para o modelo continuar corretamente
        follow_messages.append({"role": "assistant", "content": content})
        follow_messages.append(finish_instruction)

        resp, finish_reason, cont_text = call_api_fn(follow_messages, max_tokens=200)
        cont_text = (cont_text or "").strip()
        if cont_text:
            # concatena sem duplicar espaços
            if content.endswith(" ") or content.endswith("\n"):
                new_content = content + cont_text
            else:
                new_content = content.rstrip() + " " + cont_text
            # se agora parece completo, retorna
            if not _looks_truncated(new_content):
                return new_content
            # caso contrário, atualiza content e tenta novamente
            content = new_content
        else:
            # sem continuação -> retorno do que temos
            break

    return content

# ---------------------------------------------------------------------
# --- LLM summarizer enforcing PRINCIPAL prompt ---
# ---------------------------------------------------------------------
def _call_llm_summarize(question: str, quotes: List[Dict[str, str]]) -> Optional[str]:
    """
    Chama o LLM para sintetizar com base nos trechos (quotes).
    Prompt atualizado para forçar "PRINCIPAL — Resposta objetiva em 1 parágrafo".
    Retorna texto limpo ou None.
    """
    if not quotes:
        return None

    # prepara e limita os trechos para evitar estouro de tokens
    prepared = []
    for i, q in enumerate(quotes, start=1):
        src = q.get("source") or q.get("source_file") or ""
        txt = _clean_text(q.get("text") or q.get("content") or "")
        # cortar trechos muito longos para preservar tokens
        if len(txt) > 1200:
            txt = txt[:1200].rstrip() + "..."
        prepared.append({"i": i, "source": src, "text": txt})

    # --- PROMPT SISTEMA ---
    system_msg = (
        "Você é um assistente técnico e analítico que SINTETIZA respostas com base EXCLUSIVA nos trechos fornecidos. "
        "Regra principal: Responda EM PORTUGUÊS em UM PARÁGRAFO (1–3 frases), direto e objetivo. "
        "Se a resposta for 'Sim' ou 'Não', comece com 'Sim.' ou 'Não.' seguido por uma explicação curta (máx. 2 frases). "
        "Não use listas, tópicos nem exemplos longos. Mantenha a resposta entre aproximadamente 30–70 palavras; explique o suficiente sem se estender demais. "
        "Se for útil, inclua no máximo um comando ou trecho de código inline (máx. 1 linha). "
        "Se a informação estiver incompleta para uma resposta completa, termine a resposta com exatamente: \"Quer mais detalhes? (s/n)\". "
        "NÃO acrescente conhecimento externo que não esteja logicamente suportado pelos trechos abaixo. Quando mencionar algo que veio de um trecho, cite-o como [fonte #n] imediatamente após a afirmação."
    )

    # --- MENSAGEM DO USUÁRIO: contexto + trechos ---
    user_msg_intro = f"Pergunta do usuário: {question}\n\nTrechos (use apenas o que há abaixo):\n"
    for p in prepared:
        user_msg_intro += f"\n[{p['i']}] Fonte: {p['source'] or 'desconhecida'}\n{p['text']}\n"

    user_msg_instructions = (
        "\nTarefas:\n"
        "1) Responda à pergunta usando SOMENTE os trechos acima e inferências lógicas estritamente suportadas por eles.\n"
        "2) Se mencionar algo que veio de um trecho, referencie-o com [fonte #n] logo após a afirmação.\n"
        "3) Produza a resposta NO FORMATO exigido pelo sistema: 1 parágrafo (1–3 frases), começar com 'Sim.'/'Não.' quando aplicável, sem listas, 30–70 palavras, e terminar com 'Quer mais detalhes? (s/n)' caso falte informação.\n"
        "4) Inclua somente um comando/trecho de código inline se realmente útil (máx. 1 linha)."
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg_intro + "\n" + user_msg_instructions}
    ]

    # 1) chamada inicial
    try:
        resp, finish_reason, content = _call_api_with_messages(messages, max_tokens=800)
        content = (content or "").strip()
    except Exception as e:
        logging.exception("Erro na chamada inicial ao LLM em _call_llm_summarize: %s", e)
        resp, finish_reason, content = None, None, None

    if not content:
        return None

    # 2) detectar truncamento (por finish_reason ou pelo aspecto do texto) e tentar completar se necessário
    try:
        if finish_reason == "length" or _looks_truncated(content):
            # completar solicitando continuação sem repetir
            content = _complete_truncated(content, messages, _call_api_with_messages, max_retries=2)
    except Exception as e:
        logging.exception("Erro ao tentar completar resposta truncada: %s", e)

    # 3) limpeza final e retorno
    final = _clean_text(content)
    return final if final else None

# ---------------------------------------------------------------------
# --- Search helpers (vector & text) ---
# ---------------------------------------------------------------------
def _vector_search(query: str, topk: int) -> List[Dict[str, Any]]:
    vec = _embedding_or_none(query)
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
    data = _http_post(url, headers, payload)
    return data.get("value", [])

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
    data = _http_post(url, headers, payload)
    return data.get("value", [])

def _text_search(query: str,
                 topk: int,
                 force_semantic: bool = False,
                 return_raw: bool = False) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    url = f"{COG_SEARCH_ENDPOINT}/indexes/{COG_SEARCH_INDEX}/docs/search?api-version={COG_SEARCH_API_VERSION}"
    headers = {"api-key": COG_SEARCH_KEY, "Content-Type": "application/json"}

    semantic_on = ENABLE_SEMANTIC and (force_semantic or bool(COG_SEARCH_SEM_CONFIG))

    def _extract_exact_phrase_local(q: str) -> Optional[str]:
        s = (q or "").strip().lower()
        s = s[:-1] if s.endswith("?") else s
        m = re.search(r"^o que ?é?|^o que e\s+", s)
        if m:
            rest = s[m.end():].strip()
            rest = re.sub(r"^(uma|um|a|o)\s+", "", rest)
            rest = rest.strip(" :–-—")
            if 2 <= len(rest) <= 60:
                return rest
        return None

    exact_hits: List[Dict[str, Any]] = []
    exact_phrase = _extract_exact_phrase_local(query)
    if exact_phrase:
        payload_exact: Dict[str, Any] = {
            "search": f"\"{exact_phrase}\"",
            "top": min(topk, 8),
            "searchFields": SEARCH_FIELDS,
            "queryType": "simple",
            "searchMode": "all",
        }
        try:
            data_exact = _http_post(url, headers, payload_exact)
            exact_hits = data_exact.get("value", []) if isinstance(data_exact, dict) else []
        except Exception:
            exact_hits = []

    base_payload: Dict[str, Any] = {
        "search": query,
        "top": topk,
        "searchFields": SEARCH_FIELDS
    }

    if semantic_on:
        base_payload.update({
            "queryType": "semantic",
            "queryLanguage": "pt-BR",
            "answers": "extractive",
            "answersCount": 1,
            "captions": "extractive",
            "captionsHighlight": False
        })
        if COG_SEARCH_SEM_CONFIG:
            base_payload["semanticConfiguration"] = COG_SEARCH_SEM_CONFIG
    else:
        base_payload["queryType"] = "simple"

    def _do_call(payload: Dict[str, Any]) -> Dict[str, Any]:
        return _http_post(url, headers, payload)

    data: Dict[str, Any] = {}
    try:
        data = _do_call(base_payload)
    except Exception as e:
        msg = str(e)
        if "queryLanguage" in msg or "not a valid parameter for the operation 'search'" in msg:
            logging.info("semantic params rejected by service; retrying with queryType=simple")
            simple_payload = {k: v for k, v in base_payload.items()
                              if k not in ("queryType", "queryLanguage", "answers",
                                           "answersCount", "captions",
                                           "captionsHighlight", "semanticConfiguration")}
            simple_payload.update({"queryType": "simple"})
            data = _do_call(simple_payload)
        else:
            raise

    main_hits = data.get("value", []) if isinstance(data, dict) else []
    if exact_hits:
        by_id: Dict[str, Dict[str, Any]] = {}
        for h in (exact_hits + main_hits):
            hid = h.get("id") or h.get("@search.documentId")
            if hid:
                by_id[hid] = h
            else:
                by_id[f"__noid__{id(h)}"] = h
        data["value"] = list(by_id.values())

    if return_raw:
        return data
    return data.get("value", [])

# ---------------------------------------------------------------------
# --- normalization, scoring, extraction (kept from original) ---
# ---------------------------------------------------------------------
def _normalize_hit(h: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": h.get("id"),
        "score": h.get("@search.score"),
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
    STOP = {"o","a","os","as","de","do","da","dos","das","que","é","e","ou","um","uma","para","por","em","no","na","nos","nas","com","sem","se"}
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

def _extract_core_term_from_query(query: str) -> Optional[str]:
    if not query:
        return None
    s = query.strip().lower()
    if s.endswith("?"):
        s = s[:-1].strip()
    m = re.match(r"^(o que\s+(é|e))\s+", s, flags=re.IGNORECASE)
    if m:
        s = s[m.end():].strip()
    s = re.sub(r"^(uma|um|a|o)\s+", "", s, flags=re.IGNORECASE).strip()
    s = s.strip(" :–-—\"'")
    if 2 <= len(s) <= 60:
        return s
    return None

# ---------------------------------------------------------------------
# --- mini summary + answer building (kept and integrated with LLM) ---
# ---------------------------------------------------------------------
def _mini_summary_from_quotes(query: str, quotes: List[Dict[str, str]]) -> str:
    if _is_short_def_query(query):
        def_quote: Optional[str] = None
        core_term = _extract_core_term_from_query(query)
        core_term_lower = core_term.lower() if core_term else None

        if core_term_lower:
            for q in quotes:
                raw_text = q.get("text") or ""
                t_clean = _clean_text(raw_text)
                t_low = t_clean.lower()
                if (
                    t_low.startswith(core_term_lower)
                    or f"{core_term_lower} |" in t_low
                    or f"{core_term_lower}:" in t_low
                ):
                    def_quote = t_clean
                    break

        if def_quote is None:
            for q in quotes:
                t = q.get("text") or ""
                if re.search(r"(?i)\b(definiç(?:ão|ões)|o que é|:)", t) and len(t) > 40:
                    def_quote = _clean_text(t)
                    break

        if def_quote:
            core = def_quote
            if core_term_lower:
                m = re.search(rf"(?i){re.escape(core_term_lower)}\s*[\|:]\s*(.+)", def_quote)
                if m:
                    core = _clean_text(m.group(1))
            if core == def_quote:
                m = re.search(r":\s*(.+)", def_quote)
                if m:
                    core = _clean_text(m.group(1))
            if len(core) > 240:
                core = core[:220].rstrip() + "..."
            return core

    logging.debug("SYNTHESIS: sending %d quotes to LLM:", len(quotes))
    for i, q in enumerate(quotes, start=1):
        logging.debug("QUOTE %d [%s]: %.240s", i, q.get("source") or "unknown", (q.get("text") or "").replace("\n", " ")[:240])

    llm_resp = _call_llm_summarize(query, quotes)
    if llm_resp:
        llm_resp = _clean_text(llm_resp)
        if len(llm_resp) > 1000:
            llm_resp = llm_resp[:980].rstrip() + "..."
        return llm_resp

    first = _clean_text(quotes[0]["text"]) if quotes else ""
    part = first[:220].rstrip() + ("..." if len(first) > 220 else "")
    return part

def _answer_not_found() -> Dict[str, Any]:
    return {
        "text": "Não tenho essa informação",
        "bullets": [],
        "sources": [],
        "quotes": [],
        "issues": ["no_evidence"]
    }

def _find_explicit_policy_statements(nhits: List[Dict[str, Any]], query: str, keywords: Optional[List[str]] = None) -> List[Dict[str,str]]:
    """
    Busca afirmações explícitas (permissões/vedações) dentro dos documentos.
    Corrigido: usa .replace("{kw}", re.escape(kw)) ao invés de .format para evitar KeyError.
    """
    if not nhits:
        return []
    if keywords is None:
        tokens = re.findall(r"[A-Za-zÀ-ú0-9]+", (query or ""))
        keywords = [t.lower() for t in tokens if len(t) > 1]
    kws = set(keywords)

    patterns = [
        r"\b({kw})\b[\s\S]{0,120}?\b(sim|permitido|autorizad[ao]|pode ser usado|pode ser aplicad[ao]|autoriz[ae])\b",
        r"\b(sim)\b[\s\S]{0,30}?\b({kw})\b",
        r"\b({kw})\b[\s\S]{0,80}?\b(vedado|não pode|não deverão|não autorizado)\b",
        r"\b({kw})\b[\s\S]{0,120}?\b(CONFIRM|SIM|NÃO|VEDADO)\b"
    ]

    found = []
    for h in nhits:
        raw = (h.get("text") or "")
        if not raw or len(raw) < 30:
            continue
        norm = raw.lower()
        for kw in list(kws):
            # escape keyword
            kw_escaped = re.escape(kw)
            for pat in patterns:
                # use replace instead of format to avoid {} interpretation by str.format
                p = pat.replace("{kw}", kw_escaped)
                try:
                    m = re.search(p, norm, flags=re.I)
                except re.error as re_err:
                    logging.debug("invalid regex built for kw=%s pat=%s error=%s", kw, p, re_err)
                    m = None
                if m:
                    start = max(0, m.start() - 80)
                    end   = min(len(raw), m.end() + 80)
                    ctx = _clean_text(raw[start:end])
                    src = h.get("source_file") or h.get("id_original") or ""
                    if ctx and not any(ctx == x["text"] for x in found):
                        found.append({"source": src, "text": ctx})
                    break
            if found and found[-1]["source"] == (h.get("source_file") or ""):
                break
    return found

def _answer_from_quotes(query: str, quotes: List[Dict[str, str]], sources: List[str]) -> Dict[str, Any]:
    """
    Gera resposta a partir das quotes, detecta se o resumo está truncado e tenta completar
    usando o LLM; se não for possível, aplica fallback com '...' para indicar truncamento.
    """
    def _is_truncated_text(t: str) -> bool:
        if not t or not t.strip():
            return True
        t = t.rstrip()
        # se terminar com pontuação final, considerar completo
        if re.search(r'[\.!\?]["\')\]]?\s*$', t):
            return False
        # se terminar com reticências explícitas
        if t.endswith("..."):
            return False
        # heurística: se última palavra curta ou sem pontuação, considerar truncado
        m = re.search(r'([^\s]+)$', t)
        if m:
            last = m.group(1)
            if len(last) <= 3:
                return True
            if re.search(r'[^A-Za-zÀ-ú0-9\-]$', last):
                return True
            return True
        return True

    # 1) gerar summary (pode vir de mini summary -> LLM ou trecho)
    try:
        summary = _mini_summary_from_quotes(query, quotes)
    except Exception as e:
        logging.exception("Erro em _mini_summary_from_quotes: %s", e)
        summary = None

    # 2) garantir que summary exista
    if not summary:
        summary = ""

    # 3) log para debug (mostra os últimos 120 chars)
    logging.debug("SUMMARY (len=%d) tail: %.120s", len(summary or ""), (summary or "")[-120:])

    # 4) detectar truncamento; se truncado, tentar completar com LLM
    tried_completion = False
    if _is_truncated_text(summary):
        logging.info("Resumo parece truncado -> tentando completar via LLM")
        tried_completion = True
        try:
            # tenta gerar síntese completa via LLM (essa função aplica prompt PRINCIPAL)
            llm_done = _call_llm_summarize(query, quotes)
            if llm_done:
                llm_done = _clean_text(llm_done)
                logging.debug("LLM completou: tail: %.120s", llm_done[-120:])
                # se LLM retornou algo mais longo e com pontuação final, usa
                if llm_done and not _is_truncated_text(llm_done):
                    summary = llm_done
                else:
                    # se LLM também veio truncado, apenas junte (ou fallback)
                    if llm_done and len(llm_done) > len(summary):
                        summary = llm_done
        except Exception as e:
            logging.exception("Erro ao tentar completar summary com LLM: %s", e)

    # 5) se continua truncado, aplique fallback (adiciona reticências para não cortar palavra)
    if _is_truncated_text(summary):
        logging.info("Após tentativas, summary continua truncado. Aplicando fallback com reticências.")
        summary = summary.rstrip()
        if not summary.endswith("..."):
            summary = summary + "..."
    else:
        # garantir que termine com a instrução pedida quando informação incompleta
        if "Quer mais detalhes? (s/n)" not in summary:
            if not _validate_grounding(quotes):
                summary = summary.rstrip()
                if not summary.endswith(("?", "!", ".")):
                    summary = summary + "."
                summary = summary + " Quer mais detalhes? (s/n)"

    bullets = _build_bullets_from_quotes(quotes, max_bullets=3)
    return {
        "text": summary,
        "sources": sources or []
    }

# ---------------------------------------------------------------------
# --- co-occurrence, assembly and main handler (kept original logic) ---
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

        # pairwise co-occurrence fallback (keeps original behavior)
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
        char_window_size = int(os.environ.get("COOC_CHAR_WINDOW", "300"))
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
# --- main handler: optimized to parallelize vector+text searches & use embedding cache
# ---------------------------------------------------------------------
def handle_search_request(body: Dict[str, Any]) -> Dict[str, Any]:
    query = (body or {}).get("query") or ""
    topk  = int((body or {}).get("topK") or DEFAULT_TOPK)

    if _is_short_def_query(query):
        topk = max(topk, 12)

    used_embedding = False
    embedding_error = None
    hits: List[Dict[str, Any]] = []

    # helper: harvest semantic evidence (kept original)
    def _harvest_semantic_evidence_local(sem_json: Any) -> List[Dict[str, str]]:
        quotes: List[Dict[str, str]] = []
        if not isinstance(sem_json, dict):
            return quotes

        ans = sem_json.get("@search.answers") or []
        for a in ans:
            txt = (a.get("text") or "").strip()
            if not txt:
                continue
            snip = _clean_text(txt)
            if 40 <= len(snip) <= 420:
                quotes.append({"source": "", "text": snip})

        vals = sem_json.get("value") or []
        for v in vals:
            caps = v.get("@search.captions") or []
            src = v.get("source_file") or v.get("id_original") or v.get("id") or ""
            for c in caps:
                txt = (c.get("text") or "").strip()
                if not txt:
                    continue
                snip = _clean_text(txt)
                if 40 <= len(snip) <= 420:
                    quotes.append({"source": src, "text": snip})

        seen = set()
        out: List[Dict[str, str]] = []
        for q in quotes:
            t = q["text"]
            if t not in seen:
                out.append(q)
                seen.add(t)
        return out[:5]

    # --- Attempt vector and text search in parallel; use cached embedding if available ---
    vec = None
    try:
        emb_start = time.time()
        vec = _get_query_embedding(query)
        emb_dt = (time.time() - emb_start) * 1000.0
        cache_info = _cached_query_embedding_tuple.cache_info()
        logger.info("embedding time %.0fms cache_hits=%d miss=%d", emb_dt, cache_info.hits, cache_info.misses)
    except Exception as e:
        embedding_error = str(e)
        logger.warning(f"embedding retrieval failed: {e}")

    sem_json = None

    # Parallelize vector (if vec available) and text search
    with ThreadPoolExecutor(max_workers=2) as ex:
        futures = {}
        if vec:
            futures['vec'] = ex.submit(_vector_search_with_vec, vec, topk)
        else:
            # keep a fallback that does vector search (it will compute embedding) to preserve behavior
            futures['vec'] = ex.submit(_vector_search, query, topk)
        # text search runs in parallel (force_semantic based on short query detection)
        futures['text'] = ex.submit(_text_search, query, topk, force_semantic=_is_short_def_query(query), return_raw=True)

        try:
            vhits = futures['vec'].result()
            if vhits:
                used_embedding = True
                hits.extend(vhits)
                logging.info("vector search returned %d hits", len(vhits))
        except Exception as e:
            embedding_error = str(e)
            logging.warning(f"vector search failed: {e}")

        try:
            sem_json = futures['text'].result()
            thits = (sem_json.get("value", []) if isinstance(sem_json, dict) else [])
            if thits:
                # merge by id preserving existing hits
                by_id: Dict[str, Dict[str, Any]] = {}
                for h in (hits + thits):
                    hid = h.get("id")
                    if hid:
                        by_id[hid] = h
                    else:
                        by_id[f"__noid__{id(h)}"] = h
                hits = list(by_id.values())
            logging.info("text search returned")
        except Exception as e:
            logging.warning(f"text search failed: {e}")

    # normalize hits
    nhits = [_normalize_hit(h) for h in hits]
    sources = _prioritize_sources([h.get("source_file") or h.get("id_original") for h in nhits if h])

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

    q_tokens_for_search = [t for t in re.findall(r"[A-Za-zÀ-ú0-9]+", query) if len(t) > 1]
    explicit_quotes = _find_explicit_policy_statements(nhits, query, q_tokens_for_search)
    if explicit_quotes:
        existing = { _clean_text(q.get("text") or "") for q in quotes }
        new_explicit = [q for q in explicit_quotes if _clean_text(q.get("text") or "") not in existing]
        if new_explicit:
            logging.debug("FOUND EXPLICIT POLICY QUOTES: %d", len(new_explicit))
            quotes = new_explicit + quotes

    # try to add co-occurrence contexts
    quotes = _add_query_doc_cooccurrence(nhits, quotes, query, min_tokens_match=1, radius=1, max_added_docs=4)

    # scoring & sorting quotes (as original)
    if quotes:
        q_terms_counter, qset_unique, q_tokens_ordered = _make_query_keyset(query)
        scored_quotes = []
        for q in quotes:
            txt = q.get("text") or ""
            sc = _score_sentence(txt, q_terms_counter, qset_unique, q_tokens_ordered)
            scored_quotes.append((sc, q))
        for s, q in scored_quotes:
            logging.debug("QUOTE SCORE %.3f -> %.160s", s, (q.get("text") or "").replace("\n", " ")[:160])
        scored_quotes.sort(key=lambda x: x[0], reverse=True)
        quotes = [q for s, q in scored_quotes]

    # if still no quotes, fallback to an expanded semantic attempt (original behavior)
    if not quotes:
        try:
            sem_json2 = _text_search(
                query,
                max(topk, 15),
                force_semantic=True,
                return_raw=True
            )
            thits2 = (sem_json2.get("value", []) if isinstance(sem_json2, dict) else [])
            nhits2 = [_normalize_hit(h) for h in thits2] if thits2 else []
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
            logging.warning(f"second text search failed: {e}")

    # build result
    result = _answer_not_found() if not quotes else _answer_from_quotes(query, quotes, sources)

    # return full structure (keeps original fields but minimal)
    return {
        "status": "ok",
        "query": query,
        "result": result
    }

# --- main for local testing ---
if __name__ == "__main__":
    sample = {"query": "O AVCB da minha escola venceu. Posso usar o dinheiro do PDDE para renová-lo?", "topK": 8}
    out = handle_search_request(sample)
    print(json.dumps(out, indent=2, ensure_ascii=False))
