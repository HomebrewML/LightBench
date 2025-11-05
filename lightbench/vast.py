from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import httpx

DEFAULT_BASE_URL = "https://console.vast.ai/api/v0"


class VastAPIError(RuntimeError):
    pass


def _read_api_key(explicit: Optional[str]) -> str:
    if explicit:
        return explicit.strip()
    envval = os.environ.get("VAST_API_KEY", "").strip()
    if envval:
        return envval
    env_file = Path(".env")
    if env_file.exists():
        content = env_file.read_text().strip()
        if content:
            return content
    raise ValueError("Vast API key not provided. Set VAST_API_KEY or store it in .env.")


class VastClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.api_key = _read_api_key(api_key)
        self._client = httpx.Client(base_url=base_url, timeout=timeout, transport=transport)

    def close(self) -> None:
        self._client.close()

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, object]] = None,
        json: object = None,
    ) -> httpx.Response:
        query = {"api_key": self.api_key}
        if params:
            query.update(params)
        response = self._client.request(method, path, params=query, json=json)
        response.raise_for_status()
        return response

    def search_offers(
        self,
        *,
        gpu_name: Optional[str] = "RTX 4090",
        max_dph: Optional[float] = None,
        min_vram: Optional[float] = None,
        location: Optional[str] = None,
        limit: int = 20,
        order: Optional[Sequence[Sequence[str]]] = None,
        query: Optional[Mapping[str, object]] = None,
    ) -> List[Dict[str, object]]:
        if query is None:
            payload: Dict[str, object] = {
                "verified": {"eq": True},
                "external": {"eq": False},
                "rentable": {"eq": True},
                "rented": {"eq": False},
                "type": "on-demand",
                "limit": int(limit),
                "order": order or [["dlperf_per_dphtotal", "desc"]],
            }
            if gpu_name:
                payload["gpu_name"] = {"in": [gpu_name, gpu_name.replace(" ", "_")]}
            if max_dph is not None:
                payload["dph_total"] = {"lt": float(max_dph)}
            if min_vram is not None:
                payload["gpu_ram"] = {"ge": float(min_vram)}
            if location:
                payload["geolocation"] = {"eq": location}
        else:
            payload = dict(query)
        body = {"select_cols": ["*"], "q": payload}
        response = self._request("POST", "/bundles/", json=body)
        data = response.json()
        offers = data.get("offers", [])
        return list(offers) if isinstance(offers, list) else []

    def list_instances(self, *, owner: str = "me") -> List[Dict[str, object]]:
        response = self._request("GET", "/instances/", params={"owner": owner})
        data = response.json()
        instances = data.get("instances", [])
        return list(instances) if isinstance(instances, list) else []

    def create_instance(
        self,
        offer_id: int,
        *,
        image: str,
        disk_gb: int = 40,
        env: str | None = None,
        onstart: str | None = None,
        args: str | None = None,
        template_hash: str | None = None,
        bid_price: float | None = None,
        label: str | None = None,
        extra: str | None = None,
        volume_info: Mapping[str, object] | None = None,
        runtype: str = "args",
        python_utf8: bool = True,
        lang_utf8: bool = True,
        use_jupyter_lab: bool = False,
        jupyter_dir: str | None = None,
    ) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "client_id": "me",
            "image": image,
            "env": env or "",
            "price": bid_price,
            "disk": int(disk_gb),
            "label": label,
            "extra": extra,
            "onstart": onstart,
            "args": args,
            "template_hash_id": template_hash,
            "runtype": runtype,
            "python_utf8": python_utf8,
            "lang_utf8": lang_utf8,
            "use_jupyter_lab": use_jupyter_lab,
            "jupyter_dir": jupyter_dir,
            "force": False,
            "cancel_unavail": False,
        }
        if volume_info is not None:
            payload["volume_info"] = dict(volume_info)

        response = self._request("PUT", f"/asks/{offer_id}/", json=payload)
        data = response.json()
        if not data.get("success", True):
            raise VastAPIError(json.dumps(data))
        return data

    def destroy_instance(self, instance_id: int) -> Dict[str, object]:
        response = self._request("DELETE", f"/instances/{instance_id}/", json={})
        data = response.json()
        if not data.get("success", True):
            raise VastAPIError(json.dumps(data))
        return data

    def update_instance_state(self, instance_id: int, *, state: str) -> Dict[str, object]:
        response = self._request("PUT", f"/instances/{instance_id}/", json={"state": state})
        data = response.json()
        if not data.get("success", True):
            raise VastAPIError(json.dumps(data))
        return data
