from sje_model import PatientSnapshot, corrected_anion_gap, compute_sje, safe_asinh_flux


def test_corrected_anion_gap_with_albumin():
    assert round(corrected_anion_gap(140, 104, 22, 3.6), 2) == 15.0


def test_safe_asinh_flux_handles_negative_pral():
    assert safe_asinh_flux(-20, 40) < 0


def test_compute_sje_runs():
    snapshot = PatientSnapshot(
        patient_id="test",
        timestamp="2026-06-19T09:00:00Z",
        dietary_acid_load_mEq_day=65,
        net_acid_excretion_mEq_day=48,
        ph=7.37,
        baseline_ph=7.41,
        bicarbonate_mmol_l=22,
        sodium_mmol_l=140,
        chloride_mmol_l=104,
        albumin_g_dl=3.6,
        egfr_ml_min_1_73m2=72,
    )
    result = compute_sje(snapshot)
    assert result.hidden_burden_index > 0
    assert result.corrected_anion_gap_mmol_l is not None
    assert result.research_only_notice.startswith("Research prototype")
