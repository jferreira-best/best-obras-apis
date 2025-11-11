import os

SEM_CFG   = os.getenv("COG_SEARCH_SEM_CONFIG")  # "kb-semantic" se definido
API_VER   = os.getenv("COG_SEARCH_API_VERSION", "2024-07-01")
INDEX     = os.getenv("COG_SEARCH_INDEX", "kb-obras")
ENDPOINT  = os.getenv("COG_SEARCH_ENDPOINT")
KEY       = os.getenv("COG_SEARCH_KEY")
TOPK_DEF  = int(os.getenv("DEFAULT_TOPK", "8"))
