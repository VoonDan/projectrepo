"""SJE Digital Twin research prototype.

This module implements a safer SJE v1.0 scoring engine for research use only.
It is not a validated diagnostic, treatment, monitoring, or triage tool.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np


MODEL_VERSION = "sje-digital-twin-v0.1.0"


@dataclass
class WearableSnapshot:
    resting_hr_bpm: float | None = None
    hrv_ms: float | None = None
    spo2_percent: float | None = None
    sleep_hours: float | None = None
    activity_minutes: float | None = None


@dataclass
class PatientSnapshot:
    patient_id: str
    timestamp: str | datetime
    dietary_acid_load_mEq_day: float | None = None
    net_acid_excretion_mEq_day: float | None = None
    ph: float | None = None
    baseline_ph: float | None = None
    bicarbonate_mmol_l: float | None = None
    pco2_mmHg: float | None = None
    base_excess_mmol_l: float | None = None
    sodium_mmol_l: float | None = None
    chloride_mmol_l: float | None = None
    potassium_mmol_l: float | None = None
    albumin_g_dl: float | None = None
    lactate_mmol_l: float | None = None
    ketones_mmol_l: float | None = None
    urine_ph: float | None = None
    urine_ammonium_mEq_day: float | None = None
    titratable_acid_mEq_day: float | None = None
    urine_bicarbonate_mEq_day: float | None = None
    egfr_ml_min_1_73m2: float | None = None
    respiratory_rate_bpm: float | None = None
    co2_ppm: float | None = None
    wearable: dict[str, Any] | WearableSnapshot | None = None
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class SJEWeights:
    acid_flux: float = 1.0
    unmeasured_anions: float = 1.0
    buffering: float = 1.0
    ph_dynamics: float = 1.0
    respiratory: float = 0.25
    kidney_reserve: float = 0.25
    metabolic_overlay: float = 0.25


@dataclass
class SJEResult:
    model_version: str
    hidden_burden_index: float
    interpretation_band: Literal["low_research_signal", "moderate_research_signal", "high_research_signal"]
    components: dict[str, float]
    component_attribution: dict[str, float]
    corrected_anion_gap_mmol_l: float | None
    net_acid_excretion_used_mEq_day: float | None
    warnings: list[str]
    research_only_notice: str = "Research prototype only. Not validated for diagnosis, treatment, monitoring, or triage."


def corrected_anion_gap(sodium: float | None, chloride: float | None, bicarbonate: float | None, albumin: float | None) -> float | None:
    """Albumin-corrected anion gap: AGcorr = Na - Cl - HCO3 + 2.5 * (4.0 - albumin)."""
    if sodium is None or chloride is None or bicarbonate is None:
        return None
    ag = sodium - chloride - bicarbonate
    return ag if albumin is None else ag + 2.5 * (4.0 - albumin)


def net_acid_excretion_from_urine(ammonium: float | None, titratable_acid: float | None, urine_bicarbonate: float | None = None) -> float | None:
    """NAE = urine ammonium + titratable acid - urine bicarbonate."""
    if ammonium is None or titratable_acid is None:
        return None
    return max(0.0, ammonium + titratable_acid - (urine_bicarbonate or 0.0))


def safe_asinh_flux(acid_input: float | None, acid_output: float | None, scale: float = 50.0) -> float:
    """Robust flux term that handles negative PRAL and low excretion values."""
    if acid_input is None or acid_output is None:
        return 0.0
    return math.asinh((acid_input - acid_output) / scale)


def positive_excess(value: float | None, reference: float) -> float:
    if value is None:
        return 0.0
    return max(0.0, (value - reference) / reference)


def positive_deficit(value: float | None, reference: float) -> float:
    if value is None:
        return 0.0
    return max(0.0, (reference - value) / reference)


def ph_deviation_signal(ph: float | None, baseline_ph: float | None, population_baseline: float = 7.40) -> float:
    """Acid-direction pH deviation from individual or population baseline."""
    if ph is None:
        return 0.0
    baseline = baseline_ph if baseline_ph is not None else population_baseline
    return max(0.0, baseline - ph)


def respiratory_signal(pco2: float | None, respiratory_rate: float | None, co2_ppm: float | None) -> float:
    score = 0.0
    if pco2 is not None:
        score += max(0.0, (pco2 - 40.0) / 40.0)
    if respiratory_rate is not None:
        score += max(0.0, (respiratory_rate - 20.0) / 20.0)
    if co2_ppm is not None:
        score += max(0.0, (co2_ppm - 1000.0) / 1000.0)
    return score / 3.0


def kidney_reserve_signal(egfr: float | None) -> float:
    """Research proxy only. This is not a CKD diagnosis."""
    if egfr is None:
        return 0.0
    return max(0.0, (90.0 - egfr) / 90.0)


def metabolic_overlay_signal(lactate: float | None, ketones: float | None) -> float:
    score = 0.0
    if lactate is not None:
        score += max(0.0, (lactate - 2.0) / 2.0)
    if ketones is not None:
        score += max(0.0, (ketones - 0.6) / 0.6)
    return score / 2.0


def component_attribution(weighted_components: dict[str, float]) -> dict[str, float]:
    positive = {key: max(0.0, value) for key, value in weighted_components.items()}
    total = sum(positive.values())
    if total == 0:
        return {key: 0.0 for key in weighted_components}
    return {key: round(value / total, 6) for key, value in positive.items()}


def compute_sje(snapshot: PatientSnapshot, weights: SJEWeights | None = None) -> SJEResult:
    """Compute the research-only SJE hidden acid-base burden index."""
    weights = weights or SJEWeights()
    warnings: list[str] = []

    nae_from_urine = net_acid_excretion_from_urine(
        snapshot.urine_ammonium_mEq_day,
        snapshot.titratable_acid_mEq_day,
        snapshot.urine_bicarbonate_mEq_day,
    )
    nae_used = snapshot.net_acid_excretion_mEq_day if snapshot.net_acid_excretion_mEq_day is not None else nae_from_urine
    if nae_used is None:
        warnings.append("Net acid excretion missing. Acid flux component set to neutral.")

    ag_corr = corrected_anion_gap(
        snapshot.sodium_mmol_l,
        snapshot.chloride_mmol_l,
        snapshot.bicarbonate_mmol_l,
        snapshot.albumin_g_dl,
    )
    if ag_corr is None:
        warnings.append("Corrected anion gap could not be calculated. Hidden anion component set to neutral.")
    elif snapshot.albumin_g_dl is None:
        warnings.append("Albumin missing. Anion gap is uncorrected, not albumin-corrected.")

    components = {
        "acid_flux": safe_asinh_flux(snapshot.dietary_acid_load_mEq_day, nae_used),
        "unmeasured_anions": positive_excess(ag_corr, 12.0),
        "buffering": positive_deficit(snapshot.bicarbonate_mmol_l, 24.0),
        "ph_dynamics": ph_deviation_signal(snapshot.ph, snapshot.baseline_ph),
        "respiratory": respiratory_signal(snapshot.pco2_mmHg, snapshot.respiratory_rate_bpm, snapshot.co2_ppm),
        "kidney_reserve": kidney_reserve_signal(snapshot.egfr_ml_min_1_73m2),
        "metabolic_overlay": metabolic_overlay_signal(snapshot.lactate_mmol_l, snapshot.ketones_mmol_l),
    }

    weighted = {
        "acid_flux": weights.acid_flux * components["acid_flux"],
        "unmeasured_anions": weights.unmeasured_anions * components["unmeasured_anions"],
        "buffering": weights.buffering * components["buffering"],
        "ph_dynamics": weights.ph_dynamics * components["ph_dynamics"],
        "respiratory": weights.respiratory * components["respiratory"],
        "kidney_reserve": weights.kidney_reserve * components["kidney_reserve"],
        "metabolic_overlay": weights.metabolic_overlay * components["metabolic_overlay"],
    }
    hidden_burden_index = sum(weighted.values())

    if hidden_burden_index < 0.25:
        band = "low_research_signal"
    elif hidden_burden_index < 0.75:
        band = "moderate_research_signal"
    else:
        band = "high_research_signal"

    return SJEResult(
        model_version=MODEL_VERSION,
        hidden_burden_index=round(float(hidden_burden_index), 6),
        interpretation_band=band,
        components={key: round(float(value), 6) for key, value in components.items()},
        component_attribution=component_attribution(weighted),
        corrected_anion_gap_mmol_l=round(float(ag_corr), 6) if ag_corr is not None else None,
        net_acid_excretion_used_mEq_day=nae_used,
        warnings=warnings,
    )


def uncertainty_score(snapshot: PatientSnapshot, weights: SJEWeights | None = None, n_samples: int = 500, seed: int = 42) -> dict[str, float]:
    """Simple Monte Carlo uncertainty score.

    Future versions can replace this with full Bayesian posterior modelling.
    """
    rng = np.random.default_rng(seed)
    scores: list[float] = []
    for _ in range(n_samples):
        perturbed = PatientSnapshot(**asdict(snapshot))
        if perturbed.ph is not None:
            perturbed.ph = float(np.clip(rng.normal(perturbed.ph, 0.01), 6.5, 8.0))
        if perturbed.baseline_ph is not None:
            perturbed.baseline_ph = float(np.clip(rng.normal(perturbed.baseline_ph, 0.01), 6.5, 8.0))
        if perturbed.bicarbonate_mmol_l is not None:
            perturbed.bicarbonate_mmol_l = float(np.clip(rng.normal(perturbed.bicarbonate_mmol_l, 1.0), 0, 60))
        if perturbed.sodium_mmol_l is not None:
            perturbed.sodium_mmol_l = float(np.clip(rng.normal(perturbed.sodium_mmol_l, 1.0), 80, 200))
        if perturbed.chloride_mmol_l is not None:
            perturbed.chloride_mmol_l = float(np.clip(rng.normal(perturbed.chloride_mmol_l, 1.0), 50, 150))
        if perturbed.dietary_acid_load_mEq_day is not None:
            perturbed.dietary_acid_load_mEq_day = float(rng.normal(perturbed.dietary_acid_load_mEq_day, 10.0))
        if perturbed.net_acid_excretion_mEq_day is not None:
            perturbed.net_acid_excretion_mEq_day = float(max(0.0, rng.normal(perturbed.net_acid_excretion_mEq_day, 10.0)))
        scores.append(compute_sje(perturbed, weights).hidden_burden_index)

    arr = np.asarray(scores)
    return {
        "mean": round(float(np.mean(arr)), 6),
        "sd": round(float(np.std(arr)), 6),
        "p05": round(float(np.quantile(arr, 0.05)), 6),
        "p50": round(float(np.quantile(arr, 0.50)), 6),
        "p95": round(float(np.quantile(arr, 0.95)), 6),
    }


def decision_signals(snapshot: PatientSnapshot, result: SJEResult) -> list[dict[str, str]]:
    """Research-only decision layer signals."""
    signals: list[dict[str, str]] = []
    use_case = str(snapshot.context.get("use_case", "general_research"))

    if result.hidden_burden_index >= 0.75:
        signals.append({"domain": "general", "signal": "High hidden acid-base burden research signal"})
    if snapshot.egfr_ml_min_1_73m2 is not None and snapshot.egfr_ml_min_1_73m2 < 60:
        signals.append({"domain": "ckd_monitoring", "signal": "Reduced kidney reserve context present"})
    if use_case == "athlete_recovery" and snapshot.lactate_mmol_l is not None and snapshot.lactate_mmol_l > 2:
        signals.append({"domain": "athlete_recovery", "signal": "Post-exercise metabolic overlay signal"})
    if use_case == "spaceflight_monitoring" and snapshot.co2_ppm is not None and snapshot.co2_ppm > 1000:
        signals.append({"domain": "spaceflight_monitoring", "signal": "Environmental CO2 context signal"})
    if use_case == "icu_risk_detection" and result.corrected_anion_gap_mmol_l is not None and result.corrected_anion_gap_mmol_l > 16:
        signals.append({"domain": "icu_risk_detection", "signal": "Hidden anion burden research signal"})

    for signal in signals:
        signal["safety_notice"] = "Research signal only. Not a diagnosis or treatment recommendation."
    return signals


def load_snapshot(path: str | Path) -> PatientSnapshot:
    data = json.loads(Path(path).read_text())
    return PatientSnapshot(**data)


if __name__ == "__main__":
    snapshot = load_snapshot(Path(__file__).with_name("example_patient.json") if Path(__file__).with_name("example_patient.json").exists() else "example_patient.json")
    result = compute_sje(snapshot)
    print(json.dumps(asdict(result), indent=2))
    print(json.dumps({"uncertainty": uncertainty_score(snapshot), "decision_signals": decision_signals(snapshot, result)}, indent=2))
