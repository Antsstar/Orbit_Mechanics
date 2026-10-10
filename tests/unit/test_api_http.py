"""
The HTTP shell (`orbital_engine.api.http`): each route answers what its `api` function answers, an
engine refusal is a 422 carrying the same error document, and sweeps run as jobs.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Dict

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from orbital_engine import api  # noqa: E402
from orbital_engine.api import http  # noqa: E402

SCENARIO = {"builder": "two_body"}
SWEEP: Dict[str, Any] = {"scenario": SCENARIO, "horizon_s": 600.0,
                         "configs": [{"name": "kepler", "propagator": "KEPLERIAN", "dt": 60.0},
                                     {"name": "cowell", "propagator": "COWELL", "dt": 60.0,
                                      "force_models": [{"name": "point_mass_gravity"}]}]}


@pytest.fixture()
def client() -> Any:
    with TestClient(http.create_app()) as c:
        yield c


def _wait(client: TestClient, job_id: str, timeout_s: float = 120.0) -> Dict[str, Any]:
    end = time.time() + timeout_s
    while time.time() < end:
        body: Dict[str, Any] = client.get(f"/v1/jobs/{job_id}").json()
        if body["status"] in ("done", "failed", "cancelled"):
            return body
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_routes_answer_what_the_layer_answers(client: TestClient) -> None:
    assert client.get("/v1/catalog").json() == api.catalog()
    assert client.post("/v1/scenarios/describe", json=SCENARIO).json() == api.describe_scenario(SCENARIO)
    assert client.post("/v1/sweeps/validate", json=SWEEP).json() == api.validate_sweep(SWEEP)
    doc = {"scenario": SCENARIO, "dt": 60.0, "horizon_s": 300.0}
    got, want = client.post("/v1/simulate", json=doc).json(), api.simulate(doc)
    assert got["states"] == want["states"] and got["times_s"] == want["times_s"]


def test_refusals_are_422_with_paths(client: TestClient) -> None:
    r = client.post("/v1/sweeps/validate", json={**SWEEP, "horizon_s": -1})
    assert r.status_code == 422 and r.json()["errors"][0]["path"] == "horizon_s"
    r = client.post("/v1/simulate", content=b"{not json", headers={"content-type": "application/json"})
    assert r.status_code == 422 and "not JSON" in r.json()["errors"][0]["message"]
    r = client.post("/v1/jobs", json={**SWEEP, "configs": [{"name": "x", "propagator": "SECULAR_J2", "dt": 60.0}]})
    assert r.status_code == 422 and r.json()["errors"][0]["path"] == "configs[0]"     # refused before queueing


def test_a_sweep_job_runs_to_the_same_result(client: TestClient) -> None:
    r = client.post("/v1/jobs", json=SWEEP)
    assert r.status_code == 202
    body = _wait(client, r.json()["job_id"])
    assert body["status"] == "done"
    direct = api.run_sweep_request(SWEEP)
    assert [x["error"] for x in body["result"]["results"]] == [x["error"] for x in direct["results"]]
    listed = client.get("/v1/jobs").json()["jobs"]
    assert any(j["job_id"] == body["job_id"] and "result" not in j for j in listed)
    assert client.get("/v1/jobs/nope").status_code == 404


def test_queue_bound_and_cancel(monkeypatch: pytest.MonkeyPatch) -> None:
    gate = threading.Event()
    real = http.run_sweep_request

    def held(doc: Any, limits: api.Limits) -> Dict[str, Any]:
        gate.wait(30.0)
        return real(doc, limits)

    monkeypatch.setattr(http, "run_sweep_request", held)
    with TestClient(http.create_app(max_queued=1)) as c:
        first = c.post("/v1/jobs", json=SWEEP).json()["job_id"]          # running, held at the gate
        end = time.time() + 10.0
        while c.get(f"/v1/jobs/{first}").json()["status"] != "running" and time.time() < end:
            time.sleep(0.02)
        second = c.post("/v1/jobs", json=SWEEP).json()["job_id"]         # queued
        assert c.post("/v1/jobs", json=SWEEP).status_code == 429          # the queue is full
        assert c.delete(f"/v1/jobs/{first}").status_code == 409            # running: cannot cancel
        assert c.delete(f"/v1/jobs/{second}").json()["status"] == "cancelled"
        gate.set()
        assert _wait(c, first)["status"] == "done"
        assert c.get(f"/v1/jobs/{second}").json()["status"] == "cancelled"
