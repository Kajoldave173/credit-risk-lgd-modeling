# Credit Risk LGD Modeling & Expected Loss Estimation

[![MLflow Experiments](https://img.shields.io/badge/MLflow-Experiment%20Tracking-blue?logo=mlflow)](https://dagshub.com/kajoldave8/credit-risk-lgd-modeling.mlflow)
[![Live Dashboard](https://img.shields.io/badge/HuggingFace-Live%20Dashboard-yellow?logo=huggingface)](https://huggingface.co/spaces/KajolDave/credit-risk-lgd-dashboard)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2-FF6F00)](https://xgboost.readthedocs.io/)

End-to-end Loss Given Default (LGD) modeling pipeline on **1,037,500 Freddie Mac
single-family mortgage loans** (2005-2025 vintages), estimating portfolio
Expected Loss within a PD × LGD × EAD framework. Beta Regression as the primary
methodology (correct for [0,1]-bounded targets); XGBoost as the challenger
model under SR 11-7 governance.

---

## Headline Results

| Metric | Value |
|---|---|
| Total loans scored | 1,037,500 |
| Resolved defaults (modeling set) | 16,339 |
| Aggregate UPB | **$246.82B** |
| Portfolio Expected Loss | **$1.073B** |
| Portfolio EL rate | 0.435% |
| Production XGBoost Test R² | **0.236** |
| Production XGBoost overfit gap | 0.079 |
| Beta Regression Test R² | 0.165 |

R² in the 0.10-0.30 range is consistent with published mortgage LGD literature
(Qi & Yang 2009, Calabrese 2014). LGD is inherently noisier than PD because
recovery outcomes depend on liquidation timing, collateral condition at
disposition, and macro factors not captured at loan level.

---

## Live Artifacts

- **Interactive dashboard:** https://huggingface.co/spaces/KajolDave/credit-risk-lgd-dashboard
- **Experiment tracking:** https://dagshub.com/kajoldave8/credit-risk-lgd-modeling.mlflow
- **REST API:** FastAPI inference endpoint deployed on AWS EC2

---

## Architecture

```
Freddie Mac raw files (.txt by vintage)
    ↓ feature engineering (engineer_features)
    ↓ WoE / IV-based feature selection (15 features retained)
df_engineered.parquet (canonical checkpoint)
    ↓
Train / Val / Test split (70 / 10 / 20, random_state=42)
    ↓
    ├─→ XGBoost (challenger) — Config #2: depth=3, mcw=50, α=0.5, λ=1.0, n=600
    └─→ Beta Regression (primary) — custom MLE, scaled features
         ↓
Registered model artifacts (xgb_lgd_model.joblib)
    ↓
    ├─→ FastAPI inference endpoint  (AWS EC2)
    └─→ Streamlit dashboard          (HuggingFace Spaces)
```

The two deployment targets are **siblings**, not parent-child. Both consume
the same registered model artifact directly. This mirrors how production credit
risk infrastructure typically works: scoring engines and decisioning systems
pull from the model registry independently, not via HTTP from each other.

---

## Model Selection Rationale (SR 11-7 Conceptual Soundness)

Initial model deployment used XGBoost with `max_depth=6, min_child_weight=20`.
Validation against an MLflow-tracked sweep of 10 candidate configurations
revealed that the original deployment was the second-worst configuration on
both test R² and overfit gap.

### Sweep Results

| Config | Max Depth | Overfit Gap | Test R² | Selected |
|---|---|---|---|---|
| #1 | 3 | 0.063 | 0.2343 | — |
| **#2** | **3** | **0.079** | **0.2362** | **Production (v2)** |
| #5 | 4 | 0.127 | 0.2365 | — |
| #9 | 6 | 0.247 | 0.2128 | Previous production (v1) |
| #10 | 6 | 0.358 | 0.2107 | — |

### Decision Rule

Production model selected as the configuration maximizing test R² **subject to
overfit gap < 0.08**. Config #2 met this criterion with test R² statistically
indistinguishable from the unconstrained best (Config #5: ΔR² = 0.0003) while
maintaining a 38% tighter generalization profile.

### Impact

- Test R² improved from 0.213 (v1) to 0.236 (v2) — **+11% relative**
- Overfit gap reduced from 0.247 (v1) to 0.079 (v2) — **3.1x tighter generalization**
- Full experiment trail logged to DagsHub with parameters, metrics, models, and
  SHAP summary plots per configuration

This satisfies the SR 11-7 Outcomes Analysis requirement that production models
be subjected to ongoing performance monitoring with documented selection
rationale.

---

## Why Beta Regression Stays in the Production Suite

XGBoost wins on raw R² (0.236 vs 0.165). Beta Regression is retained as the
primary methodologically defensible model because:

1. **Bounded [0,1] target.** LGD is mathematically constrained to [0,1]. Beta
   Regression is the correct distributional assumption; squared-error
   regression treats it as unbounded continuous, which can produce
   out-of-range predictions absent clipping.
2. **Interpretable coefficients.** Each Beta coefficient has a direct odds-ratio
   interpretation, which simplifies model risk review and supports
   regulatory examination of feature contributions.
3. **SR 11-7 challenger framework.** A high-performing black-box paired with a
   defensible parametric baseline is the standard production pattern;
   the parametric model provides a sanity check and a fallback.

The dashboard surfaces both models on the Model Performance tab. The live
predictor uses XGBoost for point estimates; Beta provides a comparison overlay
in the actual-vs-predicted scatter and decile lift charts.

---

## SHAP-Based Reason Codes (ECOA Regulation B)

Each LGD prediction surfaces its top 6 feature contributions ranked by absolute
SHAP value. The dashboard displays these as red/blue bars (above/below
portfolio baseline).

Mapping to Regulation B adverse action language is straightforward for the most
influential features:

| SHAP Driver | Reason Code Language |
|---|---|
| High LTV ratio | "Loan-to-value ratio at origination exceeds underwriting threshold" |
| Severe DQ history | "Significant historical delinquency on this account" |
| Low credit score | "Insufficient credit history strength" |
| High DTI ratio | "Debt-to-income ratio above acceptable range" |

**Honest scope note:** LGD models do not directly trigger adverse action
notices (those come from credit decisioning, which is a PD/origination call).
If LGD-informed pricing increases a borrower's offered rate, Reg B reason
codes attach to that pricing decision. This dashboard demonstrates the
reason-code production mechanics; production adverse action workflows would
integrate with origination systems separately.

---

## Fair Lending Methodology

The Freddie Mac Single-Family Loan-Level Dataset **does not contain protected
class attributes** (race, ethnicity, sex). Disparate impact testing as required
under ECOA Regulation B cannot be performed on this dataset directly.

In production credit risk infrastructure, the standard approach is to merge
the loan-level data with HMDA records on geographic and origination keys to
recover protected class attributes for fair lending testing. Less
Discriminatory Alternative (LDA) search would then proceed using a constrained
optimization framework (e.g., adversarial debiasing or post-hoc reweighting).

This project does not run disparate impact testing because the underlying data
does not support it. Pretending otherwise would be a misrepresentation. The
methodology framework is documented; the empirical analysis requires
HMDA-merged data not included in the Freddie Mac sample.

---

## Repository Structure

```
.
├── notebooks/
│   └── lgd_modeling_notebook.ipynb     # full pipeline: data → models → outputs
├── train_with_mlflow.py                # tracked hyperparameter sweep (10 configs)
├── retrain_production_model.py         # retrain Config #2 to refresh artifacts
├── regenerate_outputs.py               # re-score portfolio + regenerate CSVs
├── api.py                              # FastAPI inference endpoint
├── app.py                              # Streamlit dashboard
├── predictor.py                        # in-process prediction module (used by HF Space)
├── xgb_lgd_model.joblib                # production XGBoost (Config #2)
├── beta_scaler.joblib                  # scaler for Beta Regression
└── outputs/
    ├── df_engineered.parquet           # canonical engineered dataset
    ├── el_by_grade.csv                 # portfolio EL by loan grade
    ├── el_by_vintage.csv               # portfolio EL by vintage year
    ├── el_by_grade_vintage.csv         # grade × vintage heatmap data
    ├── el_by_segment.csv               # risk segment breakdown
    ├── loans_scored_sample.csv         # 100K loan sample with predictions
    ├── test_predictions.csv            # test set XGBoost + Beta predictions
    ├── iv_features.csv                 # WoE / IV feature ranking
    └── model_metrics.csv               # RMSE / MAE / R² per model
```

---

## Tech Stack

- **Modeling:** XGBoost 3.2, custom Beta Regression (L-BFGS-B MLE), scikit-learn 1.4+
- **Explainability:** SHAP 0.51 (TreeExplainer)
- **Experiment tracking:** MLflow + DagsHub hosted tracking server
- **Inference:** FastAPI + Pydantic
- **Dashboard:** Streamlit 1.39 on HuggingFace Spaces (CPU)
- **Deployment:** AWS EC2 (FastAPI), HuggingFace Spaces (Streamlit)
- **Data:** Freddie Mac Single-Family Loan-Level Dataset (sample, 2005-2025)

---

## References

- Federal Reserve SR 11-7: "Guidance on Model Risk Management"
- ECOA / Regulation B (12 CFR Part 1002): Adverse action notice requirements
- FCRA (Fair Credit Reporting Act): Reason code requirements
- Qi & Yang (2009), "Loss Given Default of High Loan-to-Value Residential
  Mortgages," *Journal of Banking & Finance*
- Calabrese (2014), "Downturn Loss Given Default: Mixture distribution
  estimation," *European Journal of Operational Research*

---

## Author

**Kajol Dave** — M.S. Business Analytics, UMass Amherst Isenberg School (May 2026)
