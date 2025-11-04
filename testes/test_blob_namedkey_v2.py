import os
import re
from azure.storage.blob import BlobServiceClient
from azure.core.credentials import AzureNamedKeyCredential

API_VERSION = "2023-11-03"  # força versão estável para evitar MAC signature mismatch

def extract_name_key_from_connstr(cs: str):
    cs = (cs or "").strip().strip('"').replace("\r", "").replace("\n", "")
    m_name = re.search(r"AccountName=([^;]+)", cs, re.I)
    m_key  = re.search(r"AccountKey=([^;]+)", cs, re.I)
    if not (m_name and m_key):
        raise RuntimeError("❌ STORAGE_CONN inválida: faltam AccountName/AccountKey.")
    account_name = m_name.group(1).strip()
    account_key  = m_key.group(1).strip().replace(" ", "")
    return account_name, account_key

def main():
    # 1) Lê a connection string da env
    cs = os.getenv("STORAGE_CONN")
    if not cs:
        raise RuntimeError("❌ Defina a variável de ambiente STORAGE_CONN antes de rodar.")

    # 2) Extrai name/key (evita from_connection_string)
    account_name, account_key = extract_name_key_from_connstr(cs)
    print(f"🔐 AccountName: {account_name}")
    print(f"🔐 Tamanho da chave: {len(account_key)} caracteres")

    # 3) Cria BlobServiceClient com versão fixa
    account_url = f"https://{account_name}.blob.core.windows.net"
    cred = AzureNamedKeyCredential(account_name, account_key)
    bsc = BlobServiceClient(account_url=account_url, credential=cred, api_version=API_VERSION)

    # 4) Acessa diretamente o container "obras"
    cont = bsc.get_container_client("obras")
    if not cont.exists():
        raise RuntimeError("❌ Container 'obras' não existe (ou credencial sem acesso).")

    # 5) Lista blobs sem prefixo
    print("\n📂 Listando até 10 blobs em 'obras':")
    count = 0
    for blob in cont.list_blobs():
        print(" •", blob.name)
        count += 1
        if count >= 10:
            break
    print(f"(total parcial mostrado: {count})")

    # 6) Lista com prefixo opcional 'docs/'
    print("\n🔎 Listando até 10 blobs com prefixo 'docs/':")
    count = 0
    for blob in cont.list_blobs(name_starts_with="docs/"):
        print(" •", blob.name)
        count += 1
        if count >= 10:
            break
    if count == 0:
        print("Nenhum blob com prefixo 'docs/'.")
    else:
        print(f"(total parcial mostrado: {count})")

    print("\n✅ Teste concluído.")

if __name__ == "__main__":
    main()
