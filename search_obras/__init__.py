import logging
import json
import azure.functions as func
from typing import Any, Dict

# importa diretamente a função que você já tem
from function_app import generate_client_response

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("HTTP trigger: search_obras called")
    try:
        req_body_bytes = req.get_body()
        if not req_body_bytes:
            return func.HttpResponse(
                json.dumps({"error":"empty body","hint":"send JSON with query"}, ensure_ascii=False),
                status_code=400,
                mimetype="application/json"
            )

        body = json.loads(req_body_bytes.decode("utf-8"))
    except Exception as e:
        logging.exception("failed parsing body")
        return func.HttpResponse(
            json.dumps({"error":"invalid json","detail": str(e)}, ensure_ascii=False),
            status_code=400,
            mimetype="application/json"
        )

    # validações mínimas
    query = body.get("query") or body.get("q")
    if not query or not isinstance(query, str) or not query.strip():
        return func.HttpResponse(
            json.dumps({"error":"missing 'query' field"}, ensure_ascii=False),
            status_code=400,
            mimetype="application/json"
        )

    # normaliza topK/debug e prepara payload
    try:
        topK = int(body.get("topK") or body.get("topk") or 5)
    except Exception:
        topK = 5
    debug = bool(body.get("debug", False))

    payload = {"query": query, "topK": topK, "debug": debug, **{k:v for k,v in body.items() if k not in ("query","topK","topk","debug")}}

    # delega para sua função principal
    try:
        response_obj = generate_client_response(payload)
        return func.HttpResponse(
            json.dumps(response_obj, ensure_ascii=False),
            status_code=200,
            mimetype="application/json; charset=utf-8"
        )
    except Exception as e:
        logging.exception("internal error in search handler")
        return func.HttpResponse(
            json.dumps({"error":"internal error","detail": str(e)}, ensure_ascii=False),
            status_code=500,
            mimetype="application/json"
        )
