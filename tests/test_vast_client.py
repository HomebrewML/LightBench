from __future__ import annotations

import json

import httpx

from lightbench.vast import VastClient


def test_search_offers_builds_expected_request(monkeypatch):
    monkeypatch.setenv("VAST_API_KEY", "dummy")

    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["api_key"] = request.url.params.get("api_key")
        payload = json.loads(request.content)
        captured["payload"] = payload
        return httpx.Response(200, json={"offers": [{"id": 1}]})

    transport = httpx.MockTransport(handler)
    client = VastClient(base_url="https://example.com/api/v0", transport=transport)
    offers = client.search_offers(gpu_name="RTX 4090", max_dph=1.5, min_vram=24, limit=5)
    client.close()

    assert offers == [{"id": 1}]
    assert captured["method"] == "POST"
    assert captured["path"].endswith("/bundles/")
    assert captured["api_key"] == "dummy"
    payload = captured["payload"]
    assert payload["q"]["gpu_name"]["in"] == ["RTX 4090", "RTX_4090"]
    assert payload["q"]["limit"] == 5


def test_create_instance_payload(monkeypatch):
    monkeypatch.setenv("VAST_API_KEY", "dummy")

    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json={"success": True, "new_contract": 42})

    transport = httpx.MockTransport(handler)
    client = VastClient(base_url="https://example.com/api/v0", transport=transport)
    result = client.create_instance(
        100,
        image="example/lightbench:latest",
        disk_gb=80,
        env="-e TEST=1",
        onstart="bash",
        args="-lc 'echo hello'",
        bid_price=1.0,
        label="worker",
    )
    client.close()

    assert result["new_contract"] == 42
    assert captured["method"] == "PUT"
    assert captured["path"].endswith("/asks/100/")
    payload = captured["json"]
    assert payload["image"] == "example/lightbench:latest"
    assert payload["disk"] == 80
    assert payload["env"] == "-e TEST=1"
    assert payload["args"] == "-lc 'echo hello'"
