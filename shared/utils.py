import re
import unicodedata
from typing import List, Dict, Any
from shared.config import STOPWORDS, logger

def clean_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\r", " ")
    s = re.sub(r"\n+", " ", s)
    s = re.sub(r"\s{2,}", " ", s)
    s = re.sub(r"(\w)-\s+(\w)", r"\1\2", s)
    return s.strip()

def tokenize_pt(s: str) -> List[str]:
    s = (s or "").lower()
    s = re.sub(r"[^\wà-úÀ-ÚçÇ-]+", " ", s)
    toks = [t for t in s.split() if len(t) > 1]
    return toks

def strip_accents(s: str) -> str:
    if not s: 
        return ""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")

def normalize_hit(h: Dict[str, Any]) -> Dict[str, Any]:
    # Ajuste do score conforme lógica original
    score = (h.get("@search.rerankerScore") / 4.0) if h.get("@search.rerankerScore") else (h.get("@search.vectorSearchScore") or h.get("@search.score"))
    return {
        "id": h.get("id"),
        "score": score,
        "text": clean_text(h.get("text") or ""),
        "source_file": h.get("source_file"),
        "id_original": h.get("id_original"),
    }

def split_sentences(txt: str) -> List[str]:
    _SENT_SPLIT = re.compile(r'(?<=[\.\!\?\:])\s+')
    txt = clean_text(txt)
    return [s.strip() for s in _SENT_SPLIT.split(txt) if s.strip()]

def is_short_def_query(query: str) -> bool:
    q = clean_text(query).lower()
    toks = tokenize_pt(q)
    if len(toks) <= 6 and (
        q.startswith("o que é") or q.startswith("o que e") or
        q.startswith("defina") or q.startswith("conceito de")
    ):
        return True
    return False

def mask_secret(s: str) -> str:
    if not s:
        return "(empty)"
    s = str(s)
    if len(s) <= 8:
        return s[0:1] + "*****" + s[-1:]
    return s[:6] + "..." + s[-4:]