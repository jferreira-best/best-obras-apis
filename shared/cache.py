import time
import hashlib
import json
from collections import OrderedDict
from typing import Dict, Any, Optional
# CORREÇÃO 1: Removemos 'logger' daqui. Importamos apenas 'config'.
from shared import config 

class SimpleMemoryCache:
    def __init__(self, ttl: int, max_size: int):
        self.ttl = ttl
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()

    def _generate_key(self, body: Dict[str, Any]) -> str:
        """
        Gera uma chave única baseada nos parâmetros que alteram a resposta.
        """
        query = (body.get("query") or "").strip().lower()
        topk = body.get("topK") or config.DEFAULT_TOPK
        index = body.get("search_index") or config.COG_SEARCH_INDEX
        semantic_conf = body.get("semantic_config") or config.COG_SEARCH_SEM_CONFIG
        
        # Cria uma string única para esses parâmetros
        raw_key = f"{query}|{topk}|{index}|{semantic_conf}"
        # Retorna o hash MD5 para economizar memória na chave
        return hashlib.md5(raw_key.encode('utf-8')).hexdigest()

    def get(self, body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        key = self._generate_key(body)
        if key in self.cache:
            entry = self.cache[key]
            # Verifica se expirou (TTL)
            if time.time() < entry["expire_at"]:
                # Move para o fim (LRU - Least Recently Used)
                self.cache.move_to_end(key)
                
                # CORREÇÃO 2: Acesso correto ao logger via config
                config.logger.info(f"CACHE HIT: Pergunta recuperada da memória ({key})")
                
                return entry["data"]
            else:
                # Remove se expirou
                del self.cache[key]
        return None

    def set(self, body: Dict[str, Any], result: Dict[str, Any]):
        key = self._generate_key(body)
        # Limpeza LRU: se encheu, remove o mais antigo (primeiro item)
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
        
        self.cache[key] = {
            "data": result,
            "expire_at": time.time() + self.ttl
        }

# Instância global
request_cache = SimpleMemoryCache(ttl=config.CACHE_TTL_SECONDS, max_size=config.CACHE_MAX_SIZE)