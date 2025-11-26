import re
import os
from collections import Counter
from typing import List, Dict, Any, Optional
from shared import config, utils, clients

def call_llm_summarize(question: str, quotes: List[Dict[str, str]], compact: bool = False) -> Optional[str]:
    if not quotes:
        return None
    
    trechos_list = []
    for i, q in enumerate(quotes, 1):
        txt = (q.get("text") or q.get("content") or "").strip()
        if not txt: continue
        if len(txt) > 1200: txt = txt[:1200].rstrip() + "..."
        src = q.get("source") or q.get("source_file") or q.get("doc_title") or ""
        trechos_list.append(f"[{i}] {txt}\nFonte: {os.path.basename(src)}")

    if not trechos_list: return None
    trechos_block = "\n\n".join(trechos_list)

    system = (
        "Você é um assistente de IA focado em responder perguntas usando documentos de referência.\n"
        "Siga TODAS as instruções abaixo:\n"
        "1) Baseie sua resposta **EXCLUSIVAMENTE** nos trechos.\n"
        "2) PROIBIDO inventar fatos.\n"
        "3) CONEXÕES: Conecte o sentido da pergunta com o sentido dos trechos.\n"
        "4) RISCO vs EMERGÊNCIA: Perguntas de risco -> 'Manutenção de Risco'. Eventos ocorridos -> 'Emergência'.\n"
        "5) Se não encontrar, explique que não encontrou.\n"
        "6) Responda em português claro.\n"
    )
    user = (
        f"Pergunta: {question}\n\nTrechos:\n{trechos_block}\n\n"
        "Se os trechos trazem a informação, responda direta e concisamente."
    )

    _, _, ans = clients.call_api_with_messages(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=220 if compact else 400
    )
    return ans.strip() if ans else None

def make_query_keyset(query: str):
    q_tokens_all = utils.tokenize_pt(query)
    q_tokens = [t for t in q_tokens_all if t not in config.STOPWORDS]
    q_terms = Counter(q_tokens_all)
    qset = set(q_tokens)
    return q_terms, qset, q_tokens

def score_sentence(sent: str, q_terms: Counter, qset: Optional[set] = None, q_tokens_ordered: Optional[List[str]] = None) -> float:
    if qset is None:
        qset = {t for t in q_terms.keys() if t not in config.STOPWORDS}
    sent_clean = utils.clean_text(sent).lower()
    toks = [t for t in utils.tokenize_pt(sent) if t not in config.STOPWORDS]
    if not toks: return 0.0
    
    matched = qset & set(toks)
    presence = len(matched) / max(1, len(qset))
    overlap = sum(q_terms.get(t, 0) for t in toks) / (1.0 + (len(toks) ** 0.5))
    
    ngram_matches = 0
    if q_tokens_ordered:
        for n in (2, 3):
            if len(q_tokens_ordered) < n: continue
            for i in range(len(q_tokens_ordered) - n + 1):
                gram = " ".join(q_tokens_ordered[i:i+n])
                if re.search(r'\b' + re.escape(gram.lower()) + r'\b', sent_clean, flags=re.I):
                    ngram_matches += 1
    
    bigram_boost = min(1.2, 0.45 * ngram_matches)
    score = (2.2 * presence) + (1.0 * overlap) + bigram_boost
    return float(score)

def find_explicit_policy_statements(nhits: List[Dict[str, Any]], query: str, q_tokens: List[str]) -> List[Dict[str, str]]:
    if not nhits: return []
    qset = {t.lower() for t in q_tokens if t and t.lower() not in config.STOPWORDS}
    out = []
    for h in nhits:
        raw = h.get("text") or ""
        sents = utils.split_sentences(raw)
        for s in sents:
            ss = utils.clean_text(s)
            low = ss.lower()
            if any(k in low for k in config._NORMATIVE_KEYWORDS):
                toks = {t for t in utils.tokenize_pt(low) if t not in config.STOPWORDS}
                if qset and (qset & toks):
                    out.append({"source": h.get("source_file") or h.get("id_original") or "", "text": ss})
    
    seen = set(); final = []
    for q in out:
        if q["text"] not in seen:
            seen.add(q["text"])
            final.append(q)
        if len(final) >= 6: break
    return final

def extract_quotes(hits: List[Dict[str, Any]], query: str, per_doc: int = 3, max_quotes: int = 8) -> List[Dict[str, str]]:
    q_terms, qset, q_tokens = make_query_keyset(query)
    quotes = []
    seen = set()

    # (Lógica simplificada para brevidade - mantendo a essência do score)
    for h in hits:
        src = h.get("source_file") or h.get("id_original") or ""
        sents = utils.split_sentences(h.get("text") or "")
        scored = []
        for i, s in enumerate(sents):
            sc = score_sentence(s, q_terms, qset, q_tokens)
            scored.append((i, sc))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        taken = 0
        for i, sc in scored:
            if sc <= 0: break
            snip = utils.clean_text(sents[i]) # Simplificando window
            if len(snip) < 20 or snip in seen: continue
            quotes.append({"source": src, "text": snip})
            seen.add(snip)
            taken += 1
            if taken >= per_doc or len(quotes) >= max_quotes: break
        if len(quotes) >= max_quotes: break
    return quotes

def derive_query_key_tokens(query: str) -> List[str]:
    toks = re.findall(r"[A-Za-zÀ-ú0-9]+", query or "")
    out = []
    for t in toks:
        tl = utils.clean_text(t.strip())
        if tl and tl.lower() not in config.STOPWORDS and len(tl) >= 2:
            out.append(tl)
    return list(dict.fromkeys(out))

def add_query_doc_cooccurrence(nhits, quotes, query, max_added_docs=4):
    # Wrapper simplificado para manter compatibilidade
    # A lógica completa de co-ocorrência pode ser movida para cá
    # Para o exemplo, vamos retornar os quotes originais se não implementarmos toda a complexidade
    return quotes