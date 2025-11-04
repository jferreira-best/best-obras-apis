import azure.functions as func
from shared.telemetry import log_event


app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)


@app.function_name(name="webhook_travamento")
@app.route(route="webhooks/travamento", methods=["POST"])  # POST /api/webhooks/travamento
def webhook_travamento(req: func.HttpRequest) -> func.HttpResponse:
    payload = req.get_json()
    log_event("travamento_webhook", payload or {})
    # TODO: publicar em Service Bus / abrir ticket
    return func.HttpResponse(status_code=202, body="ok")
