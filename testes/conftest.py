import os
from azure.storage.blob import BlobServiceClient
cs = os.getenv("STORAGE_CONN")
print("Conn len:", len(cs) if cs else None)
bsc = BlobServiceClient.from_connection_string(cs)
cont = bsc.get_container_client("obras")
print("Exemplo de 3 blobs:", [b.name for b in cont.list_blobs(name_starts_with="")][:3])