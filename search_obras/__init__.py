# search_obras/__init__.py
import json
import logging
import azure.functions as func

from shared import function_app as fa

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("search_obras HTTP trigger received")

    try:
        body = req.get_json()
    except ValueError:
        body = {k: v for k, v in req.params.items()}

    try:
        # Chama o handler no shared/function_app.py
        # Ajuste o nome abaixo se o seu handler tiver outro nome.
        result = fa.handle_search_request(body)

        if isinstance(result, func.HttpResponse):
            return result

        return func.HttpResponse(
            json.dumps(result, ensure_ascii=False),
            status_code=200,
            mimetype="application/json"
        )
    except Exception as e:
        logging.exception("Erro processando search_obras")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )
