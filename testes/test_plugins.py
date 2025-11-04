import requests_mock
from sk.plugins.sed_certificados.CertificadosSkill import CertificadosSkill
from sk.plugins.sala_do_futuro.GovBrSkill import GovBrSkill


APIM = "http://apim.test"


def test_certificados_status():
    with requests_mock.Mocker() as m:
        m.get(f"{APIM}/api/certificados/status/123", json={"status": "aprovado"})
        s = CertificadosSkill()
        out = s.status("123")
        assert out["status"] == "aprovado"


def test_govbr_status():
    with requests_mock.Mocker() as m:
        m.get(f"{APIM}/api/govbr/status/111", json={"bloqueado": False})
        s = GovBrSkill()
        out = s.check_status("111")
        assert out["bloqueado"] is False
