import azure.functions as func
from sk.plugins.plataformas.ConectividadeSkill import ConectividadeSkill
from shared.telemetry import log_event


kernel = build_kernel()


# registra plugins
kernel.add_plugin(GovBrSkill(), "govbr")
kernel.add_plugin(FallbackAutosaveSkill(), "autosave")
kernel.add_plugin(CertificadosSkill(), "certificados")
kernel.add_plugin(SEDLogsSkill(), "sedlogs")
kernel.add_plugin(SincronizacaoPlataformasSkill(), "plataforma")
kernel.add_plugin(ConectividadeSkill(), "net")


app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)


@app.function_name(name="orchestrate")
@app.route(route="orchestrate", methods=["POST"]) # POST /api/orchestrate
def orchestrate(req: func.HttpRequest) -> func.HttpResponse:
data = req.get_json()
intent = (data or {}).get("intent")
inputs = (data or {}).get("inputs", {})


log_event("orchestrate_in", {"intent": intent})


try:
if intent == "verificar_certificado":
cpf = inputs["cpf"]
res = kernel.plugins["certificados"].status(cpf)
return func.HttpResponse(status_code=200, body=str(res))


if intent == "diagnosticar_sed":
cpf = inputs["cpf"]
res = kernel.plugins["sedlogs"].get_last_action(cpf)
return func.HttpResponse(status_code=200, body=str(res))


if intent == "reenviar_autosave":
chave = inputs["chave"]
res = kernel.plugins["autosave"].reenviar(chave)
return func.HttpResponse(status_code=200, body=str(res))


if intent == "status_sincronizacao":
ra = inputs["ra"]
res = kernel.plugins["plataforma"].get_status(ra)
return func.HttpResponse(status_code=200, body=str(res))


if intent == "ping":
res = kernel.plugins["net"].ping()
return func.HttpResponse(status_code=200, body=str(res))


return func.HttpResponse("Intent não suportada", status_code=400)


except Exception as e:
log_event("orchestrate_error", {"intent": intent, "err": str(e)})
return func.HttpResponse(f"erro: {e}", status_code=500)