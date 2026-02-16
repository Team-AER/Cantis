#!/usr/bin/env python3
"""Phase 1 local API server scaffold for Auralux.

This server mirrors the expected REST contract used by the Swift app:
- GET  /health
- POST /generate
- GET  /jobs/<id>
- POST /jobs/<id>/cancel

Replace fake generation in `_run_job` with ACE-Step MLX inference calls.
"""

from __future__ import annotations

import argparse
import json
import random
import threading
import time
import uuid
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Optional


@dataclass
class Job:
    id: str
    status: str = "queued"
    progress: float = 0.0
    message: Optional[str] = None
    audio_path: Optional[str] = None
    cancelled: bool = False


class JobStore:
    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self) -> Job:
        job = Job(id=str(uuid.uuid4()))
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **kwargs: object) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            for key, value in kwargs.items():
                setattr(job, key, value)


JOBS = JobStore()


def _run_job(job_id: str, prompt: str, duration: float) -> None:
    JOBS.update(job_id, status="running", progress=0.05)

    steps = 20
    for step in range(1, steps + 1):
        job = JOBS.get(job_id)
        if not job:
            return
        if job.cancelled:
            JOBS.update(job_id, status="cancelled", message="Job cancelled")
            return

        time.sleep(0.2)
        jitter = random.uniform(-0.01, 0.01)
        JOBS.update(job_id, progress=max(0.0, min(0.95, step / steps + jitter)))

    # Placeholder output file path (replace with actual generated file).
    output_dir = Path.home() / "Library" / "Application Support" / "Auralux" / "Generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{job_id}.wav"

    # Stub file allows end-to-end wiring in UI and history.
    output.write_bytes(b"RIFF0000WAVEfmt ")

    JOBS.update(
        job_id,
        status="completed",
        progress=1.0,
        message=f"Generated track for: {prompt[:64]}",
        audio_path=str(output),
    )


class Handler(BaseHTTPRequestHandler):
    server_version = "AuraluxServer/0.1"

    def _read_json(self) -> Dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _send_json(self, payload: Dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send_json({"status": "ok"})
            return

        if self.path.startswith("/jobs/"):
            job_id = self.path.split("/")[2]
            job = JOBS.get(job_id)
            if not job:
                self._send_json({"error": "job not found"}, HTTPStatus.NOT_FOUND)
                return
            self._send_json(
                {
                    "jobID": job.id,
                    "status": job.status,
                    "progress": job.progress,
                    "message": job.message,
                    "audioPath": job.audio_path,
                }
            )
            return

        self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/generate":
            payload = self._read_json()
            prompt = str(payload.get("prompt", ""))
            duration = float(payload.get("duration", 30))

            job = JOBS.create()
            worker = threading.Thread(target=_run_job, args=(job.id, prompt, duration), daemon=True)
            worker.start()

            self._send_json({"jobID": job.id, "status": "queued", "message": "accepted"}, HTTPStatus.ACCEPTED)
            return

        if self.path.startswith("/jobs/") and self.path.endswith("/cancel"):
            job_id = self.path.split("/")[2]
            job = JOBS.get(job_id)
            if not job:
                self._send_json({"error": "job not found"}, HTTPStatus.NOT_FOUND)
                return
            JOBS.update(job_id, cancelled=True)
            self._send_json({"jobID": job.id, "status": "cancelling"})
            return

        self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Auralux API server listening on http://127.0.0.1:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
