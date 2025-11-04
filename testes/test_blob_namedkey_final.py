import os, base64
from azure.storage.blob import BlobServiceClient
from azure.core.credentials import AzureNamedKeyCredential

API_VERSION = "2023-11-03"  # versão estável

def must_get_env(k: str) -> str:
    v = os.getenv(k)
    if not v:
        raise RuntimeError(f"Variável {k} não definida.")
    return v

def main():
    account_name = must_get_env("AZURE_ACCOUNT_NAME").strip()
    account_key  = must_get_env("AZURE_ACCOUNT_KEY").strip().replace(" ", "").strip('"')

    # Sanity check da chave (Base64 decodável e múltiplo de 4)
    if len(account_key) % 4 != 0:
        raise RuntimeError(f"AccountKey com tamanho inesperado ({len(account_key)}). Parece truncada.")
    try:
        raw = base64.b64decode(account_key, validate=True)
    except Exception as e:
        raise RuntimeError(f"AccountKey inválida (Base64): {e}")

    print(f"🔐 AccountName: {account_name}")
    print(f"🔐 AccountKey len: {len(account_key)} chars | bytes decodados: {len(raw)}")

    cred = AzureNamedKeyCredential(account_name, account_key)
    bsc  = BlobServiceClient(account_url=f"https://{account_name}.blob.core.windows.net",
                             credential=cred,
                             api_version=API_VERSION)

    cont = bsc.get_container_client("obras")
    if not cont.exists():
        raise RuntimeError("Container 'obras' não existe (ou credencial sem acesso).")

    print("\n📂 Primeiros blobs em 'obras':")
    n = 0
    for b in cont.list_blobs():
        print(" •", b.name)
        n += 1
        if n >= 10: break
    if n == 0:
        print(" (nenhum blob listado)")

    print("\n🔎 Prefixo 'docs/':")
    n = 0
    for b in cont.list_blobs(name_starts_with="docs/"):
        print(" •", b.name)
        n += 1
        if n >= 10: break
    if n == 0:
        print(" (nenhum blob com prefixo 'docs/')")

    print("\n✅ Teste OK.")

if __name__ == "__main__":
    main()
