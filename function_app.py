import azure.functions as func
import logging
import json
from shared.search_service import handle_search_request

app = func.FunctionApp()

@app.function_name(name="search_obras")
@app.route(route="search_obras", auth_level=func.AuthLevel.ANONYMOUS)
def http_search_trigger(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Recebendo requisição de busca (Modular).')
    
    try:
        try:
            body = req.get_json()
        except ValueError:
            return func.HttpResponse(
                json.dumps({"error": "Invalid JSON body"}, ensure_ascii=False),
                mimetype="application/json", status_code=400
            )

        result = handle_search_request(body)

        return func.HttpResponse(
            json.dumps(result, ensure_ascii=False),
            mimetype="application/json", status_code=200
        )

    except Exception as e:
        logging.exception(f"Erro crítico na function: {e}")
        return func.HttpResponse(
            json.dumps({"error": f"Erro interno: {str(e)}"}, ensure_ascii=False),
            mimetype="application/json", status_code=500
        )