import os
import logging

# Configuração de Log
logger = logging.getLogger("function_app_shared")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL))


# Configuração de Cache
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600")) # 1 hora padrão
CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "1000"))       # Máximo de 1000 perguntas guardadas

# --- Helpers de ENV ---
def _safe_str_env(key: str, default: str) -> str:
    return os.environ.get(key, default)

def _safe_int_env(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        return default

# --- Variáveis de Ambiente ---
COG_SEARCH_ENDPOINT    = _safe_str_env("COG_SEARCH_ENDPOINT", "").rstrip("/")
COG_SEARCH_KEY         = _safe_str_env("COG_SEARCH_KEY", "")
COG_SEARCH_INDEX       = _safe_str_env("COG_SEARCH_INDEX", "")
COG_SEARCH_API_VERSION = _safe_str_env("COG_SEARCH_API_VERSION", "2024-07-01")
COG_SEARCH_SEM_CONFIG  = _safe_str_env("COG_SEARCH_SEM_CONFIG", "")

SEARCH_FIELDS          = _safe_str_env("SEARCH_FIELDS", "doc_title,text")
DEFAULT_TOPK           = _safe_int_env("DEFAULT_TOPK", 6)
ENABLE_SEMANTIC        = _safe_str_env("ENABLE_SEMANTIC", "true").lower() in ("1","true","yes","on")

AOAI_ENDPOINT          = _safe_str_env("AOAI_ENDPOINT", "").rstrip("/")
AOAI_API_KEY           = _safe_str_env("AOAI_API_KEY", "")
AOAI_EMB_DEPLOYMENT    = _safe_str_env("AOAI_EMB_DEPLOYMENT", "")
AOAI_CHAT_DEPLOYMENT   = _safe_str_env("AOAI_CHAT_DEPLOYMENT", "")
AOAI_API_VERSION       = _safe_str_env("AOAI_API_VERSION", "2023-10-01")

OPENAI_API_KEY         = _safe_str_env("OPENAI_API_KEY", "")
OPENAI_MODEL           = _safe_str_env("OPENAI_MODEL", "gpt-4o-mini")

EMBED_DIM              = _safe_int_env("EMBED_DIM", 3072)
HTTP_TIMEOUT_SHORT     = _safe_int_env("HTTP_TIMEOUT_SHORT", 8)
HTTP_TIMEOUT_LONG      = _safe_int_env("HTTP_TIMEOUT_LONG", 20)

# Guardrails
RELEVANCE_THRESHOLD_HITS = float(os.getenv("RELEVANCE_THRESHOLD_HITS", "0.10"))
MIN_QUOTES_REQUIRED      = int(os.getenv("MIN_QUOTES_REQUIRED", "2"))
ALLOW_COMPLETION_WHEN_WEAK = os.getenv("ALLOW_COMPLETION_WHEN_WEAK", "true").lower() in ("1","true","yes","on")

# --- Listas e Constantes ---
STOPWORDS = {
    "o","a","os","as","de","do","da","dos","das","que",
    "é","e","ou","um","uma","para","por","em","no","na",
    "nos","nas","com","sem","se","ao","à","às","aos","não",
    "qual","quais","como","onde","quando"
}

_DOMAIN_KEYWORDS_RAW = [
    "obra","obras","manutencao","seguranca","engenharia","projeto",
    "canteiro","nr","manual","procedimento","instalacao","elevador","incendio",
    "aquecimento","chuva","demanda","inspecao","checklist","construcao","servicos",
    "agua","potavel","nao potavel","qualidade","reservatorio","termico","coletores","solar","hidraulica"
]
DOMAIN_KEYWORDS = [k.lower() for k in _DOMAIN_KEYWORDS_RAW]

_NORMATIVE_KEYWORDS = [
    "procedimento", "procedimentos", "norma", "normas", "política", "políticas",
    "deve", "devem", "obrigatório", "obrigatória", "obrigatórios",
    "responsável", "responsáveis", "responsabilidade",
    "é necessário", "é proibido", "não é permitido", "requisitos", "diretriz", "diretrizes"
]