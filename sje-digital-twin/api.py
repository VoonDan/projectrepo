"""FastAPI wrapper for the SJE Digital Twin research prototype."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from sje_model import PatientSnapshot, SJEWeights, compute_sje, decision_signals, uncertainty_score

app = FastAPI(
    title="SJE Digital Twin API",
    version="0.1.0",
    description="Research-only API for Saini-Jesslyn Equation digital twin scoring.",
)


class ScoreRequest(BaseModel):
    snapshot: dict[str, Any]
    weights: dict[str, float] | None = None
    include_uncertainty: bool = True


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "sje-digital-twin", "mode": "research-only"}


@app.post("/score")
def score(request: ScoreRequest) -> dict[str, Any]:
    snapshot = PatientSnapshot(**request.snapshot)
    weights = SJEWeights(**request.weights) if request.weights else None
    result = compute_sje(snapshot, weights)
    uncertainty = uncertainty_score(snapshot, weights) if request.include_uncertainty else None
    return {
        "result": asdict(result),
        "uncertainty": uncertainty,
        "decision_signals": decision_signals(snapshot, result),
    }
