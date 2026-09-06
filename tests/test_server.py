import json
import threading
from types import SimpleNamespace

import torch
from fastapi.testclient import TestClient

import neodisco.server as server


class FakeRunner:
    def __init__(self, tmp_path):
        self.out_dir = tmp_path
        self.runtime_overrides = {"device": "cpu", "precision": "fp32"}
        self.submitted = []
        self.jobs = {}
        self.order = []
        self.lock = threading.Lock()

    def submit(self, settings):
        self.submitted.append(settings)
        return SimpleNamespace(public=lambda: {
            "id": "test", "state": "queued", "seed": settings["seed"],
            "width": settings["width"], "height": settings["height"],
        })


def test_api_validates_before_submit_and_imports_disco_json(tmp_path):
    runner = FakeRunner(tmp_path)
    app = server.build_app(runner, tmp_path)
    client = TestClient(app)
    invalid = client.post("/api/generate", json={"prompt_text": "x", "width": 65})
    assert invalid.status_code == 400
    assert not runner.submitted
    old = json.dumps({"text_prompts": {"0": ["warm:2", "cold:-0.5"]}, "ViTB32": True})
    valid = client.post("/api/generate", json={
        "disco_json": old, "width": 64, "height": 64, "steps": 2,
        "skip_steps": 0, "cut_overview": 1, "cut_innercut": 0,
    })
    assert valid.status_code == 200
    assert runner.submitted[0]["clip_models"] == ["ViTB32"]
    assert runner.submitted[0]["weights"] == [4 / 3, -1 / 3]


def test_api_rejects_web_batch(tmp_path):
    client = TestClient(server.build_app(FakeRunner(tmp_path), tmp_path))
    response = client.post("/api/generate", json={"prompt_text": "x", "batch_size": 2})
    assert response.status_code == 400


def test_api_rejects_late_reference_mode(tmp_path):
    client = TestClient(server.build_app(FakeRunner(tmp_path), tmp_path))
    response = client.post("/api/generate", json={
        "prompt_text": "x", "deterministic": True})
    assert response.status_code == 400
    assert "restart neodisco-web" in response.json()["detail"]


def test_clip_bank_cache_is_bounded(monkeypatch, tmp_path):
    created = []

    class FakeBank:
        def __init__(self, names, device):
            created.append(tuple(names))

    monkeypatch.setattr(server, "ClipBank", FakeBank)
    runner = server.Runner(tmp_path, tmp_path / "out", start_worker=False)
    runner._bank(["ViTB32"], torch.device("cpu"))
    runner._bank(["RN50"], torch.device("cpu"))
    assert len(runner._banks) == 1
    assert len(created) == 2
