import logging
import re
import os  # <--- ADICIONADO para manipular caminhos de arquivo
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List

# Imports dos módulos compartilhados
from shared import config, utils, clients, rag_logic
from shared.cache import request_cache

def _prioritize_sources(sources: List[str], limit: int = 4) -> List[str]:
    """
    Extrai apenas o nome do arquivo (ex: Manual.pdf), remove duplicatas 
    preservando a ordem de relevância e limita a quantidade.
    """
    clean_list = []
    seen = set()
    
    for s in sources:
        if not s:
            continue
        
        # 1. Normaliza barras (garante funcionamento em Windows e Linux)
        s_norm = s.replace("\\", "/")
        
        # 2. Pega apenas o nome do arquivo (remove C:/Users/Temp/...)
        filename = os.path.basename(s_norm)
        
        # 3. Deduplicação pelo nome do arquivo
        if filename not in seen:
            clean_list.append(filename)
            seen.add(filename)
            
    return clean_list[:limit]

def _filter_hits_by_relevance(nhits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filtra resultados baseados no threshold definido no config."""
    if not nhits:
        return []
    
    strong = [h for h in nhits if (h.get("score") or 0.0) >= config.RELEVANCE_THRESHOLD_HITS]
    
    # Se tiver hits fortes, retorna eles. Senão, retorna os top 5 por score para tentar salvar a resposta.
    if strong:
        return strong
    
    return sorted(nhits, key=lambda x: (x.get("score") or 0.0), reverse=True)[:5]

def handle_search_request(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Orquestrador principal da busca (Controller).
    Gerencia Cache -> Threads de Busca -> Lógica RAG -> LLM -> Resposta.
    """
    
    # 1. VERIFICAÇÃO DE CACHE
    # Se debug=True, ignoramos o cache para forçar o processamento real.
    force_refresh = body.get("debug", False)
    
    if not force_refresh:
        cached_result = request_cache.get(body)
        if cached_result:
            return cached_result

    # 2. PARSE DOS PARÂMETROS
    query = (body or {}).get("query") or ""
    topk = int((body or {}).get("topK") or config.DEFAULT_TOPK)
    compact = bool((body or {}).get("compact", False))
    
    req_index = (body or {}).get("search_index") or config.COG_SEARCH_INDEX
    req_sem_cfg = (body or {}).get("semantic_config") or config.COG_SEARCH_SEM_CONFIG

    if not req_index:
        raise ValueError("Parâmetro 'search_index' é obrigatório no JSON da requisição.")

    # 3. REGRA DE NEGÓCIO (ELEVADOR)
    # Se a query falar de "preso" ou "trancado" em "elevador", forçamos termos de emergência.
    query_lower = query.lower()
    if ("preso" in query_lower or "trancado" in query_lower) and "elevador" in query_lower:
        logging.info("Regra de negócio acionada: Forçando contexto de emergência para elevador.")
        query = query + " emergência risco"

    # Ajusta TopK se for pergunta de definição curta (precisa de mais contexto para acertar)
    if utils.is_short_def_query(query):
        topk = max(topk, 12)

    # 4. EXECUÇÃO PARALELA (Busca Vetorial + Busca Texto/Semântica)
    hits: List[Dict[str, Any]] = []
    sem_json = None

    with ThreadPoolExecutor(max_workers=2) as ex:
        # Dispara busca vetorial
        future_vec = ex.submit(clients.vector_search, query, topk, req_index)
        
        # Dispara busca textual (híbrida/semântica)
        # Nota: Passamos force_semantic=True se for pergunta curta "O que é..."
        future_text = ex.submit(
            clients.text_search, 
            query, 
            topk, 
            req_sem_cfg, 
            req_index, 
            force_semantic=utils.is_short_def_query(query), 
            return_raw=True
        )
        
        # Coleta resultados Texto
        try:
            sem_json = future_text.result(timeout=config.HTTP_TIMEOUT_LONG + 5)
            thits = sem_json.get("value", []) if isinstance(sem_json, dict) else []
        except Exception as e:
            logging.warning("Text search failed: %s", e)
            thits = []
            sem_json = {}

        # Coleta resultados Vetor
        try:
            vhits = future_vec.result(timeout=config.HTTP_TIMEOUT_SHORT + 10)
        except Exception as e:
            logging.warning("Vector search failed: %s", e)
            vhits = []

        # Merge dos resultados (Deduplicação por ID)
        hits_map = {}
        # Prioridade para Vetorial na lista, mas o dicionário remove duplicatas
        for h in (vhits + thits):
            hid = h.get("id") or f"_noid_{id(h)}"
            if hid not in hits_map:
                hits_map[hid] = h
        hits = list(hits_map.values())

    # 5. NORMALIZAÇÃO E FILTRAGEM
    nhits = [utils.normalize_hit(h) for h in hits]
    nhits = _filter_hits_by_relevance(nhits)

    # 6. EXTRAÇÃO DE EVIDÊNCIAS (RAG)
    # Extrai trechos relevantes baseados em score de sentença
    quotes = rag_logic.extract_quotes(nhits, query)
    
    # Busca Normativa Específica (procura por "deve", "proibido", "norma")
    q_tokens_search = [t for t in re.findall(r"[A-Za-zÀ-ú0-9]+", query) if len(t) > 1]
    explicit = rag_logic.find_explicit_policy_statements(nhits, query, q_tokens_search)
    
    # Mescla as evidências normativas com as gerais (dando prioridade às normativas)
    seen_txt = {q["text"] for q in quotes}
    for e in explicit:
        if e["text"] not in seen_txt:
            quotes.insert(0, e) 
            seen_txt.add(e["text"])

    # (Opcional) Co-ocorrência poderia ser chamada aqui se implementada completamente
    # quotes = rag_logic.add_query_doc_cooccurrence(nhits, quotes, query)

    # 7. GERAÇÃO DA RESPOSTA (LLM)
    result: Dict[str, Any] = {}

    if not quotes:
        result = {
            "text": "Não encontrei informações sobre esse tema nos documentos indexados.", 
            "sources": []
        }
    else:
        # Chama o LLM para sumarizar os quotes
        summary = rag_logic.call_llm_summarize(query, quotes, compact=compact)
        sources = _prioritize_sources([q.get("source") for q in quotes])
        
        if not summary or summary == "NAO_ENCONTRADO":
             result = {
                 "text": "Não encontrei informações específicas nos documentos, apesar de ter localizado termos relacionados.", 
                 "sources": []
             }
        else:
             result = {"text": summary, "sources": sources}

    # Monta o objeto final
    final_response = {
        "status": "ok",
        "versao": "v2.2-modular-clean-sources",
        "query": query,
        "result": result
    }

    # 8. SALVAR NO CACHE
    request_cache.set(body, final_response)

    return final_response