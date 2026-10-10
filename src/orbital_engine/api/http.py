"""
An HTTP service over the JSON layer (`orbital_engine.api`): FastAPI, the `[api]` extra.

    <env>/python.exe -m orbital_engine.api.http [--host 127.0.0.1] [--port 8000]

Interactive documentation is served at `/docs`. Every route is a shell over one `api` function, and
every engine refusal is the same `ApiError` document with HTTP 422:

| Route | Body | Answer |
|---|---|---|
| `GET /v1/catalog` | - | `api.catalog()` |
| `POST /v1/scenarios/describe` | scenario | `api.describe_scenario` |
| `POST /v1/sweeps/validate` | sweep | `api.validate_sweep` (warnings) |
| `POST /v1/simulate` | simulate | `api.simulate`, synchronous, capped by `Limits` |
| `POST /v1/jobs` | sweep | 202 `{"job_id"}`; validated first, then queued |
| `GET /v1/jobs` / `GET /v1/jobs/{id}` | - | status: `queued`, `running`, `done` (+ `result`), `failed` (+ `errors`), `cancelled` |
| `DELETE /v1/jobs/{id}` | - | cancels a job that has not started; a running sweep cannot be interrupted |

**Why jobs.** A sweep computes truth and runs every configuration; that can take minutes, longer
than a request should be held open. Jobs run on a small worker pool (`workers`, default 1, because a
sweep's timings are only comparable when it is not competing for the CPU). The queue is bounded
(`max_queued`): a full queue answers 429 rather than accepting work it will not reach. Finished
jobs are kept in memory, the oldest dropped beyond `keep_finished`; nothing persists across a
restart.

**Local by default.** The service binds to `127.0.0.1` and has no authentication: it is a local tool
for a UI or an agent on the same machine. Exposing it beyond that needs authentication and tighter
`Limits` in front of it first.
"""
from __future__ import annotations

import argparse
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .catalog import catalog
from .requests import API_VERSION, ApiError
from .service import Limits, describe_scenario, run_sweep_request, simulate, validate_sweep

__all__ = ["create_app", "JobStore", "main"]

JsonDict = Dict[str, Any]


@dataclass
class Job:
    id: str
    request: Any
    status: str = "queued"
    submitted: float = field(default_factory=time.time)
    started: Optional[float] = None
    finished: Optional[float] = None
    result: Optional[JsonDict] = None
    errors: Optional[List[Dict[str, str]]] = None
    future: Optional["Future[None]"] = None

    def to_json(self, with_result: bool = True) -> JsonDict:
        out: JsonDict = {"api_version": API_VERSION, "job_id": self.id, "status": self.status,
                         "submitted": self.submitted, "started": self.started, "finished": self.finished}
        if self.errors is not None:
            out["errors"] = self.errors
        if with_result and self.result is not None:
            out["result"] = self.result
        return out


class JobStore:
    """Sweep jobs on a worker pool, with a bounded queue and a bounded memory of finished jobs."""

    def __init__(self, limits: Limits, workers: int = 1, max_queued: int = 16, keep_finished: int = 100) -> None:
        self.limits = limits
        self.max_queued = max_queued
        self.keep_finished = keep_finished
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="sweep")
        self._jobs: "OrderedDict[str, Job]" = OrderedDict()
        self._lock = threading.Lock()

    def submit(self, doc: Any) -> Job:
        validate_sweep(doc, self.limits)                       # refuse a bad request now, not in the queue
        with self._lock:
            queued = sum(1 for j in self._jobs.values() if j.status == "queued")
            if queued >= self.max_queued:
                raise HTTPException(status_code=429, detail=f"{queued} sweeps already queued; try again later")
            job = Job(id=uuid.uuid4().hex, request=doc)
            self._jobs[job.id] = job
            job.future = self._pool.submit(self._run, job)
            self._trim()
        return job

    def _run(self, job: Job) -> None:
        with self._lock:
            if job.status != "queued":
                return
            job.status, job.started = "running", time.time()
        payload: Optional[JsonDict] = None
        errors: Optional[List[Dict[str, str]]] = None
        try:
            payload = run_sweep_request(job.request, self.limits)
            status = "done"
        except ApiError as exc:
            status, errors = "failed", exc.errors
        except Exception as exc:                                       # the job, not the service, fails
            status, errors = "failed", [{"path": "", "message": f"{type(exc).__name__}: {exc}"}]
        with self._lock:
            job.status, job.result, job.errors, job.finished = status, payload, errors, time.time()

    def _trim(self) -> None:
        finished = [k for k, j in self._jobs.items() if j.status in ("done", "failed", "cancelled")]
        for key in finished[: max(0, len(finished) - self.keep_finished)]:
            del self._jobs[key]

    def get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"no job {job_id!r}")
        return job

    def all(self) -> List[Job]:
        with self._lock:
            return list(self._jobs.values())

    def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        with self._lock:
            if job.status != "queued":
                raise HTTPException(status_code=409, detail=f"job is {job.status}; only a queued job can be cancelled")
            job.status, job.finished = "cancelled", time.time()
        return job

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


async def _body(request: Request) -> Any:
    try:
        return await request.json()
    except ValueError as exc:
        raise ApiError([{"path": "", "message": f"the body is not JSON: {exc}"}]) from exc


def create_app(limits: Limits = Limits(), workers: int = 1, max_queued: int = 16, keep_finished: int = 100
               ) -> FastAPI:
    jobs = JobStore(limits, workers, max_queued, keep_finished)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        jobs.shutdown()

    app = FastAPI(lifespan=lifespan, title="OrbitalEngine", version=str(API_VERSION),
                  description="A model-fidelity comparison engine: one scenario under many model "
                              "configurations, diffed against a common truth. Documents are described in "
                              "`orbital_engine.api.requests`; GET /v1/catalog lists what they may name.")
    app.state.jobs = jobs

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(status_code=422, content=exc.to_json())

    @app.get("/v1/catalog")
    def get_catalog() -> JsonDict:
        return catalog()

    @app.post("/v1/scenarios/describe")
    async def post_describe(request: Request) -> JsonDict:
        return describe_scenario(await _body(request), limits)

    @app.post("/v1/sweeps/validate")
    async def post_validate(request: Request) -> JsonDict:
        return validate_sweep(await _body(request), limits)

    @app.post("/v1/simulate")
    async def post_simulate(request: Request) -> JsonDict:
        return simulate(await _body(request), limits)

    @app.post("/v1/jobs", status_code=202)
    async def post_job(request: Request) -> JsonDict:
        return jobs.submit(await _body(request)).to_json()

    @app.get("/v1/jobs")
    def list_jobs() -> JsonDict:
        return {"api_version": API_VERSION, "jobs": [j.to_json(with_result=False) for j in jobs.all()]}

    @app.get("/v1/jobs/{job_id}")
    def get_job(job_id: str) -> JsonDict:
        return jobs.get(job_id).to_json()

    @app.delete("/v1/jobs/{job_id}")
    def delete_job(job_id: str) -> JsonDict:
        return jobs.cancel(job_id).to_json()

    return app


def main(argv: Optional[List[str]] = None) -> None:
    import uvicorn
    parser = argparse.ArgumentParser(description="Serve the OrbitalEngine HTTP API.")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: local only)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=1, help="concurrent sweep jobs")
    args = parser.parse_args(argv)
    uvicorn.run(create_app(workers=args.workers), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
