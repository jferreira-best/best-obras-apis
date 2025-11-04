#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import requests
from urllib.parse import urljoin

API_VERSION = "2024-07-01"

def get(url, headers, what):
    try:
        r = requests.get(url, headers=headers, timeout=20)
        ok = r.status_code == 200
        print(f"[{what}] HTTP {r.status_code}")
        if ok:
            # Mostra um resumo curto do JSON
            data = r.json()
            # imprime só as chaves de topo para não poluir
            print(f"[{what}] OK. Top-level keys: {list(data.keys())[:8]}")
        else:
            print(f"[{what}] ERRO: {r.text}")
        return ok
    except Exception as e:
        print(f"[{what}] EXCEPTION: {e}")
        return False

def main():
    ap = argparse.ArgumentParser(description="Teste de conexão com Azure AI Search")
    ap.add_argument("--search-endpoint", required=True, help="endpoint")
    ap.add_argument("--search-api-key", required=True, help="API-Key")
    ap.add_argument("--api-version", default=API_VERSION)
    args = ap.parse_args()

    base = args.search_endpoint.rstrip("/") + "/"
    headers = {"api-key": args.search_api_key}

    print("== Azure AI Search connectivity test ==")
    print(f"Endpoint : {base}")
    print(f"API ver. : {args.api_version}")

    # 1) service stats (saúde/limites do serviço)
    url_stats = f"{base}servicestats?api-version={args.api_version}"
    ok1 = get(url_stats, headers, "servicestats")

    # 2) lista de índices
    url_idx = f"{base}indexes?api-version={args.api_version}"
    ok2 = get(url_idx, headers, "list_indexes")

    if ok1 and ok2:
        print("\n✅ Conseguiu conectar e consultar o serviço e listar índices.")
        print("   (Chave parece ser ADMIN key válida para esse endpoint.)")
    else:
        print("\n❌ Falha em pelo menos uma chamada.")
        print("   Verifique:")
        print("   - Se o endpoint está correto (https://<seu-servico>.search.windows.net)")
        print("   - Se a chave é a ADMIN KEY (não use query key)")
        print("   - Se o serviço está ativo e sem restrições de rede/Firewall")
        print("   - Se a versão da API é suportada pelo serviço")

if __name__ == "__main__":
    main()
