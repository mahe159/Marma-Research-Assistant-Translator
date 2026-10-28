from fastapi.testclient import TestClient

import main

client = TestClient(main.app)


def test_root_reports_online():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "online"


def test_empty_text_is_rejected():
    r = client.post(
        "/translate",
        json={"text": "   ", "source_lang": "English", "target_lang": "Bangla"},
    )
    assert r.status_code == 400


def test_missing_field_is_rejected():
    r = client.post("/translate", json={"text": "hello"})
    assert r.status_code == 422


def test_translate_returns_model_output(monkeypatch):
    calls = {}

    def fake_generate(text, source_lang, target_lang):
        calls.update(text=text, source=source_lang, target=target_lang)
        return "translated"

    monkeypatch.setattr(main, "generate_translation", fake_generate)
    r = client.post(
        "/translate",
        json={"text": "Hello", "source_lang": "English", "target_lang": "Bangla"},
    )
    assert r.status_code == 200
    assert r.json() == {
        "translated_text": "translated",
        "source_lang": "English",
        "target_lang": "Bangla",
    }
    assert calls == {"text": "Hello", "source": "English", "target": "Bangla"}


def test_inference_exception_becomes_500(monkeypatch):
    def boom(**_):
        raise RuntimeError("model not loaded")

    monkeypatch.setattr(main, "generate_translation", boom)
    r = client.post(
        "/translate",
        json={"text": "Hello", "source_lang": "English", "target_lang": "Bangla"},
    )
    assert r.status_code == 500
    assert "model not loaded" in r.json()["detail"]
