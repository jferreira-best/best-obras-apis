# app.py
# ---------------------------------------------------------------
# API FastAPI para Busca Semântica + QA sobre Azure AI Search
# ---------------------------------------------------------------

import os, json, requests
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

# -------- CONFIGURAÇÃO --------
AZURE_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
EMB_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMB_DEPLOYMENT", "text-embedding-3-small")
CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini")

SEARCH_ENDPOINT = os.environ["SEARCH_ENDPOINT"]
SEARCH_API_KEY = os.environ["SEARCH_API_KEY"]
SEARCH_INDEX = os.getenv("SEARCH_INDEX", "kb-pain-points")

client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_ENDPOINT,
)

# -------- MODELOS DE DADOS --------
class Filters(BaseModel):
    assunto: Optional[str] = "seduc"
    area_interesse: Optional[str] = "conhecimento"

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    filters: Optional[Filters] = Filters()

class QARequest(SearchRequest):
    pass

class SearchResult(BaseModel):
    id: str
    score: float
    text: Optional[str]
    metadata: Dict[str, Optional[str]]

class SearchResponse(BaseModel):
    count: int
    results: List[SearchResult]

class QAResponse(BaseModel):
    answer: str
    sources: List[Dict[str, str]]

# -------- FUNÇÕES BASE --------
def embed_query(query: str) -> List[float]:
    resp = client.embeddings.create(input=[query], model=EMB_DEPLOYMENT)
    return resp.data[0].embedding

def search_azure(query: str, vector: List[float], top_k: int, f: Filters) -> List[Dict]:
    headers = {"Content-Type": "application/json", "api-key": SEARCH_API_KEY}
    body = {
        "search": query,
        "queryType": "semantic",
        "vectorQueries": [
            {"kind": "vector", "vector": vector, "fields": "content_vector", "k": top_k}
        ],
        "filter": f"assunto eq '{f.assunto}' and area_interesse eq '{f.area_interesse}'",
        "select": "id,text,area,dor_n1,dor_n2,descricao,impacto,causas,script,prioridade,persona",
        "top": top_k
    }
    url = f"{SEARCH_ENDPOINT}/indexes/{SEARCH_INDEX}/docs/search?api-version=2024-07-01"
    r = requests.post(url, headers=headers, json=body)
    if r.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Erro Search: {r.status_code} {r.text}")
    return r.json().get("value", [])

def build_context(hits: List[Dict]) -> str:
    parts = []
    for i, h in enumerate(hits, 1):
        parts.append(
            f"[DOC#{i}] {h.get('dor_n1')} › {h.get('dor_n2')} ({h.get('area')})\n"
            f"Descrição: {h.get('descricao')}\n"
            f"Causas: {h.get('causas')}\n"
            f"Impacto: {h.get('impacto')}\n"
            f"Ações: {h.get('script')}\n"
        )
    return "\n".join(parts)

def synthesize_answer(question: str, context: str) -> str:
    prompt = f"""
Pergunta do usuário:
{question}

Documentos relevantes:
{context}

Instruções:
- Gere uma resposta estruturada com:
  1) Resumo
  2) Causas prováveis
  3) Impacto
  4) Ações recomendadas
  5) Quem deve agir (se aplicável)
- Use apenas informações dos documentos.
- Escreva em português claro e direto, sem floreios.
"""
    resp = client.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        temperature=0.2,
        messages=[
            {"role": "system", "content": "Você é um analista especialista em processos educacionais e operacionais."},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content.strip()

# -------- FASTAPI APP --------
app = FastAPI(title="Busca Semântica SEDUC", version="1.0.0")

@app.get("/")
def home():
    return {"status": "ok", "routes": ["/search", "/qa"]}

@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    vec = embed_query(req.query)
    hits = search_azure(req.query, vec, req.top_k, req.filters)
    results = [
        SearchResult(
            id=h["id"],
            score=h.get("@search.score", 0),
            text=h.get("text"),
            metadata={
                "area": h.get("area"),
                "dor_n1": h.get("dor_n1"),
                "dor_n2": h.get("dor_n2"),
                "prioridade": h.get("prioridade"),
                "persona": h.get("persona"),
            },
        )
        for h in hits
    ]
    return SearchResponse(count=len(results), results=results)

@app.post("/qa", response_model=QAResponse)
def qa(req: QARequest):
    vec = embed_query(req.query)
    hits = search_azure(req.query, vec, req.top_k, req.filters)
    if not hits:
        return QAResponse(answer="Não encontrei resultados relevantes.", sources=[])
    context = build_context(hits)
    answer = synthesize_answer(req.query, context)
    sources = [
        {"id": h["id"], "dor_n1": h.get("dor_n1"), "dor_n2": h.get("dor_n2"), "area": h.get("area")}
        for h in hits[:3]
    ]
    return QAResponse(answer=answer, sources=sources)
