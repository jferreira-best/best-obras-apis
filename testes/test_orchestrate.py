import json
from orchestrate.__init__ import orchestrate
import types


class _Req:
    def __init__(self, payload):
        self._j = payload

    def get_json(self):
        return self._j


class _Resp:
    def __init__(self, status_code: int, body: str):
        self.status_code = status_code
        self._body = body

    def get_body(self):
        return self._body


# Monkeypatch de plugins para não chamar rede
from orchestrate.__init__ import kernel
kernel.plugins["certificados"].status = lambda cpf: {"status": "ok", "cpf": cpf}


def test_orchestrate_verificar_certificado():
    req = _Req({"intent": "verificar_certificado", "inputs": {"cpf": "123"}})
    resp = orchestrate(req)
    assert resp.status_code == 200
