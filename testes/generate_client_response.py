#!/usr/bin/env python3
"""
generate_client_response.py (prompt AOAI ajustado para formal/conciso em pt-BR)

Substitua o arquivo anterior por este. Mantém:
 - geração de embedding AOAI
 - busca vetorial Azure Cognitive Search
 - deduplicação
 - tentativa de recuperar documento por id
 - reranking + geração final via AOAI Chat (com prompt formal/conciso pt-BR)
 - salvamento de arquivos: final_ai_output.json, short.txt, step_by_step.txt, email.txt, chatbot_payload.json, evidence.txt
"""
import os
import sys
import json
import requests
from typing import List, Dict, Any
from urllib.parse import quote

# -------------------------
# Helpers env
# -------------------------
def get_env(name: str, default: Any = None, required: bool = False):
    v = os.environ.get(name, default)
    if required and (v is None or str(v) == ""):
        raise EnvironmentError(f"Missing required environment variable: {name}")
    return v

# -------------------------
# Load config from env
# -------------------------
SEARCH_ENDPOINT = get_env("SEARCH_ENDPOINT", required=True)
SEARCH_API_KEY = get_env("SEARCH_API_KEY", required=True)
SEARCH_INDEX = get_env("SEARCH_INDEX", "kb-obras")
SEARCH_PROFILE = get_env("SEARCH_PROFILE", "vprofile")
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

# -------------------------
# AOAI Embeddings
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

# -------------------------
# Azure Cognitive Search vector query
# -------------------------
def search_vector(embedding: List[float], topk: int = DEFAULT_TOPK) -> Dict[str, Any]:
    topk = min(int(topk), MAX_TOPK)
    url = SEARCH_ENDPOINT.rstrip("/") + f"/indexes/{SEARCH_INDEX}/docs/search?api-version={SEARCH_API_VERSION}"
    headers = {"api-key": SEARCH_API_KEY, "Content-Type": "application/json"}
    search_body = {
        "count": True,
        # don't include @search.score in select
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
    resp = requests.post(url, headers=headers, json=search_body, timeout=REQUEST_TIMEOUT)
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"Azure Search error: {resp.status_code} {resp.text}")
    return resp.json()

# -------------------------
# Retrieve doc by id (GET)
# -------------------------
def get_doc_by_id(doc_id: str) -> Dict[str, Any]:
    encoded = quote(doc_id, safe='')
    url = SEARCH_ENDPOINT.rstrip("/") + f"/indexes/{SEARCH_INDEX}/docs/{encoded}?api-version={SEARCH_API_VERSION}"
    headers = {"api-key": SEARCH_API_KEY}
    resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    if resp.status_code == 200:
        return resp.json()
    return {"_fetch_error_status": resp.status_code, "_fetch_error_text": resp.text}

# -------------------------
# Dedupe
# -------------------------
def dedupe_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for r in results:
        doc_id = r.get("id") or r.get("key") or r.get("@search.documentId")
        text = r.get("text") or r.get("content") or ""
        fingerprint = doc_id if doc_id else text[:200]
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        out.append(r)
    return out

# -------------------------
# AOAI Chat call (rerank + generate)
# -------------------------
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
# Build prompts for reranking/generation (adjusted: formal, concise, pt-BR)
# -------------------------
def build_rerank_prompt(query: str, docs: List[Dict[str, Any]]) -> (str, str):
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
        title = d.get("doc_title") or d.get("title") or "<no-title>"
        source = d.get("source_file") or d.get("source") or "<no-source>"
        snippet = (d.get("text") or "")[:1000].strip().replace("\n", " ")
        pieces.append(f"{i}) id: {doc_id}\n title: {title}\n source: {source}\n snippet: {snippet}\n")
    pieces.append(
        "Task: Com base SOMENTE nas evidências acima, gere o JSON descrito. Não inclua texto fora do JSON. Se faltar informação, use 'informação insuficiente'."
    )
    user = "\n\n".join(pieces)
    return system, user

# -------------------------
# Robust helpers (file writing / convert)
# -------------------------
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

def save_file(path: str, content: Any):
    with open(path, "w", encoding="utf-8") as f:
        f.write(to_str_safe(content))

# -------------------------
# Main
# -------------------------
def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:]).strip()
    else:
        query = input("Digite a sua pergunta/consulta para gerar embedding e consultar o index: ").strip()
    if not query:
        print("Query vazia. Abortando.")
        sys.exit(1)

    #print(f"[1] Gerando embedding para: {query!r}")
    embedding = get_embedding(query)
    #print(f"[OK] Embedding len={len(embedding)}. Fazendo busca vetorial...")

    search_json = search_vector(embedding, topk=DEFAULT_TOPK)
    hits = search_json.get("value") or search_json.get("results") or []
    #print(f"[2] Hits brutos: {len(hits)} (count: {search_json.get('@odata.count')})")

    hits_u = dedupe_results(hits)
    #print(f"[3] Após dedup: {len(hits_u)}")
    if len(hits_u) == 0:
        print("Nenhum resultado relevante.")
        sys.exit(0)

    top_k = hits_u[:3]  # top 3 evidence
    full_docs = []
    for r in top_k:
        doc_id = r.get("id") or r.get("key")
        if doc_id:
            doc_full = get_doc_by_id(doc_id)
            if isinstance(doc_full, dict) and doc_full.get("id"):
                combined = {**r, **doc_full}
                full_docs.append(combined)
            else:
                r["_fetch_error"] = doc_full
                full_docs.append(r)
        else:
            full_docs.append(r)

    # Save evidence snippets
    evidence_lines = []
    for i, d in enumerate(full_docs, start=1):
        evidence_lines.append(f"=== Documento #{i} ===")
        evidence_lines.append(f"id: {d.get('id')}")
        evidence_lines.append(f"title: {d.get('doc_title') or d.get('title')}")
        evidence_lines.append(f"source: {d.get('source_file') or d.get('source')}")
        evidence_lines.append(f"score: {d.get('@search.score')}")
        evidence_lines.append("snippet (primeiros 5000 chars):")
        evidence_lines.append((d.get("text") or d.get("content") or json.dumps(d))[:5000])
        evidence_lines.append("\n")
    save_file("evidence.txt", "\n".join(evidence_lines))

    # Reranking + generation using AOAI Chat
    #print("[4] Executando reranking + geração final via AOAI Chat (pt-BR, formal)...")
    system_prompt, user_prompt = build_rerank_prompt(query, full_docs)
    chat_resp = call_aoai_chat(system_prompt, user_prompt)

    # Extract model content
    try:
        choices = chat_resp.get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content") or choices[0].get("text") or ""
        else:
            content = chat_resp.get("text") or json.dumps(chat_resp, ensure_ascii=False)
    except Exception:
        content = json.dumps(chat_resp, ensure_ascii=False)

    # Parse JSON robustly (strip fences, find first JSON object)
    final_output = {}
    try:
        stripped = content.strip()
        if stripped.startswith("```"):
            parts = stripped.split("```")
            for p in parts:
                p = p.strip()
                if p.startswith("{") or p.startswith("["):
                    stripped = p
                    break
        idx = stripped.find("{")
        if idx != -1:
            candidate = stripped[idx:]
            final_output = json.loads(candidate)
        else:
            final_output = {"text": content}
    except Exception:
        final_output = {"text": content}

    # Normalize if list or missing fields
    if isinstance(final_output, list):
        final_output = {"text": "\n".join([to_str_safe(x) for x in final_output])}

    if not isinstance(final_output, dict) or (("short" not in final_output) and ("text" not in final_output)):
        # fallback simple generator using snippets
        primary = full_docs[0]
        primary_snippet = (primary.get("text") or primary.get("content") or "")[:800]
        short_msg = (
            "Para abrir um chamado de urgência, entre em contato com a equipe responsável pelo atendimento (telefone ou e-mail). "
            "Forneça título 'Chamado URGENTE', unidade/escola, descrição curta e contato."
        )
        step_msg = [
            "Identifique o canal de atendimento (telefone, e-mail ou portal).",
            "Reúna informações essenciais: unidade, CPF/RA (se aplicável), descrição e prints.",
            "Abra o chamado, anexe evidências e guarde o número do protocolo."
        ]
        email_msg = (
            f"Assunto: Abertura de Chamado URGENTE\n\nPrezados,\n\nSolicito a abertura de chamado com PRIORIDADE URGENTE.\n\n"
            f"Resumo do problema (evidência): {primary_snippet[:400]}...\n\nAtenciosamente,\n[Seu nome] — [setor / telefone]"
        )
        final_output = {
            "short": short_msg,
            "step_by_step": step_msg,
            "email": email_msg,
            "confidence": 0.5,
            "sources": [{"id": d.get("id"), "title": d.get("doc_title"), "source": d.get("source_file")} for d in full_docs],
            "next_steps": "Abrir chamado no portal e encaminhar protocolo ao time.",
            "call_to_action": "Envie escola, CPF/RA e telefone para que eu possa abrir o chamado."
        }

    # Save outputs safely
    save_file("final_ai_output.json", json.dumps(final_output, ensure_ascii=False, indent=2))
    if "short" in final_output:
        save_file("short.txt", final_output.get("short"))
    elif "text" in final_output:
        save_file("short.txt", final_output.get("text"))

    if "step_by_step" in final_output:
        save_file("step_by_step.txt", final_output.get("step_by_step"))

    if "email" in final_output:
        save_file("email.txt", final_output.get("email"))

    # chatbot payload
    docs_short = [{"id": d.get("id"), "title": d.get("doc_title") or d.get("title"), "source": d.get("source_file") or d.get("source")} for d in full_docs]
    chatbot_payload = {
        "user_message": query,
        "bot_reply_short": final_output.get("short") or final_output.get("text") or "",
        "show_sources_button": {"label": "Ver evidências", "docs": docs_short}
    }
    save_file("chatbot_payload.json", json.dumps(chatbot_payload, ensure_ascii=False, indent=2))

    # Print summary
    #print("\n=== FINAL OUTPUT (final_ai_output.json) ===\n")
    print(json.dumps(final_output, ensure_ascii=False, indent=2))
    print("\nArquivos gerados: final_ai_output.json, short.txt, step_by_step.txt, email.txt, chatbot_payload.json, evidence.txt")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERRO] {e}")
        sys.exit(1)
