# SJE Digital Twin Architecture

Research implementation of the **Saini-Jesslyn Equation (SJE)** as a four-layer digital twin for hidden acid-base burden modelling.

> Research prototype only. Not validated for diagnosis, treatment, monitoring, triage, or medical decision-making.

## Four-layer architecture

```mermaid
flowchart LR
    subgraph L1[Input Layer]
        A1[Diet logs]
        A2[Food-image AI]
        A3[Blood gas]
        A4[Electrolytes + albumin]
        A5[Lactate + ketones]
        A6[Urine pH + ammonium + titratable acid]
        A7[eGFR]
        A8[Respiratory rate + CO2 exposure]
        A9[Wearables]
    end

    subgraph L2[Physiology Layer]
        B1[Acid production]
        B2[Renal excretion]
        B3[Respiratory compensation]
        B4[Extracellular buffering]
        B5[Skeletal buffering]
        B6[Intracellular buffering]
    end

    subgraph L3[AI / Model Layer]
        C1[SJE v1.0 score]
        C2[Bayesian-style uncertainty]
        C3[Mixed-effects model hooks]
        C4[Explainable attribution]
        C5[Patient-specific baselines]
    end

    subgraph L4[Clinical Decision Layer]
        D1[Preventive ageing]
        D2[CKD monitoring]
        D3[Athlete recovery]
        D4[ICU risk detection]
        D5[Spaceflight monitoring]
    end

    L1 --> L2 --> L3 --> L4
```

## Safer SJE v1.0 computational form

The original conceptual form uses a logarithmic flux ratio. This implementation uses a safer acid-input/output mismatch term:

```text
asinh((dietary_acid_load_mEq_day - net_acid_excretion_mEq_day) / 50)
```

This handles plant-rich diets where PRAL may be negative and avoids invalid logarithms.

## Files

- `sje_model.py` — core SJE scoring engine, physiology helpers, uncertainty scoring, and decision signals.
- `api.py` — FastAPI wrapper.
- `example_patient.json` — synthetic example input.
- `requirements.txt` — minimal dependencies.

## Run locally

```bash
pip install -r requirements.txt
python sje_model.py
uvicorn api:app --reload
```

API docs:

```text
http://127.0.0.1:8000/docs
```

## Development roadmap

- Add FHIR adapters.
- Add MLflow model registry.
- Add PyMC Bayesian posterior model.
- Add longitudinal patient baseline store.
- Add MIMIC-IV validation notebook.
- Add SHAP explanations after outcome model training.
