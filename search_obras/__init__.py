import json
import logging
import azure.functions as func

# Importe a lógica principal
from shared import function_app as fa

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger (search_obras) processed a request.')

    try:
        body = req.get_json()
    except ValueError:
        body = {k: v for k, v in req.params.items()}
        if not body:
            body = {} # Garante que 'body' seja um dict

    try:
        # --- ESTA É A CHAMADA PRINCIPAL ---
        result = fa.handle_search_request(body)
        # ----------------------------------

        if isinstance(result, func.HttpResponse):
            return result

        return func.HttpResponse(
             json.dumps(result, ensure_ascii=False),
             mimetype="application/json",
             status_code=200
        )

    except Exception as e:
        # --- ESTA É A CORREÇÃO CRÍTICA ---
        # Se a lógica falhar, NÓS PEGAMOS O ERRO E O LOGAMOS.
        error_message = f"Unhandled exception in handle_search_request: {e}"
        logging.exception(error_message) # Loga o stack trace completo

        return func.HttpResponse(
             json.dumps({"status": "error", "message": error_message}),
             mimetype="application/json",
             status_code=500 # Retorna 500, mas com a mensagem de erro
        )