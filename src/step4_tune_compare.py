"""
Step 4: Tune with Optuna (scored by Qini, 5-fold CV), compare
T-learner vs S-learner vs simple heuristic, then test ONCE.
Run from the project root:  python src/step4_tune_compare.py
(Takes a few minutes.)
"""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold

from uplift_utils import extra_buyers_at, qini_coefficient, qini_curve

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

PROC_DIR = Path("data/processed")
MODEL_DIR = Path("models")
OUT_DIR = Path("outputs")
MODEL_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

SEED = 42
N_TRIALS = 30  # lower to 10 if your computer is slow

# ---------- 1. Load ----------
train = pd.read_csv(PROC_DIR / "train.csv").reset_index(drop=True)
test = pd.read_csv(PROC_DIR / "test.csv").reset_index(drop=True)
with open(PROC_DIR / "feature_cols.json") as f:
    feature_cols = json.load(f)

# Folds keep the same share of (treatment, conversion) combinations
strata = train["treatment"].astype(str) + "_" + train["conversion"].astype(str)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
folds = list(skf.split(train, strata))


# ---------- 2. Three ways to produce an uplift score ----------
def build_model(params):
    return LGBMClassifier(
        **params, subsample_freq=1, random_state=SEED, verbose=-1
    )


def t_learner_scores(params, tr, va):
    """Two models: one on emailed customers, one on non-emailed."""
    m_t = build_model(params).fit(
        tr.loc[tr.treatment == 1, feature_cols],
        tr.loc[tr.treatment == 1, "conversion"],
    )
    m_c = build_model(params).fit(
        tr.loc[tr.treatment == 0, feature_cols],
        tr.loc[tr.treatment == 0, "conversion"],
    )
    return (
        m_t.predict_proba(va[feature_cols])[:, 1]
        - m_c.predict_proba(va[feature_cols])[:, 1]
    )


def s_learner_scores(params, tr, va):
    """One model with 'treatment' as an extra column."""
    cols = feature_cols + ["treatment"]
    m = build_model(params).fit(tr[cols], tr["conversion"])
    p1 = m.predict_proba(va[feature_cols].assign(treatment=1))[:, 1]
    p0 = m.predict_proba(va[feature_cols].assign(treatment=0))[:, 1]
    return p1 - p0


def heuristic_scores(params, tr, va):
    """No ML: higher past spend = higher score."""
    return va["history"].to_numpy()


# ---------- 3. Cross-validated Qini ----------
def cv_qini(score_fn, params):
    scores = []
    for tr_idx, va_idx in folds:
        tr, va = train.iloc[tr_idx], train.iloc[va_idx]
        s = score_fn(params, tr, va)
        scores.append(
            qini_coefficient(va["conversion"].to_numpy(), va["treatment"].to_numpy(), s)
        )
    return float(np.mean(scores)), float(np.std(scores))


def sample_params(trial):
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 50, 400),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        num_leaves=trial.suggest_int("num_leaves", 2, 16),
        min_child_samples=trial.suggest_int("min_child_samples", 50, 500),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1.0, 50.0, log=True),
    )


# Step 3 settings, as the starting point
baseline_params = dict(
    n_estimators=200,
    learning_rate=0.03,
    num_leaves=8,
    min_child_samples=100,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=5.0,
)

print("Cross-validated Qini (mean +/- std over 5 folds):")
cv_results = {}
cv_results["Heuristic (past spend)"] = cv_qini(heuristic_scores, None)
cv_results["T-learner (Step 3 params)"] = cv_qini(t_learner_scores, baseline_params)
cv_results["S-learner (Step 3 params)"] = cv_qini(s_learner_scores, baseline_params)
for k, (m, s) in cv_results.items():
    print(f"  {k:28s} {m:7.3f} +/- {s:.3f}")

# ---------- 4. Optuna tuning ----------
def tune(score_fn, name):
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED)
    )
    study.optimize(
        lambda trial: cv_qini(score_fn, sample_params(trial))[0],
        n_trials=N_TRIALS,
    )
    best = study.best_params
    mean, std = cv_qini(score_fn, best)
    print(f"\nTuned {name}: CV Qini = {mean:.3f} +/- {std:.3f}")
    print("  best params:", {k: round(v, 4) if isinstance(v, float) else v for k, v in best.items()})
    return best, (mean, std)


print("\nTuning with Optuna ...")
best_t, cv_results["T-learner (tuned)"] = tune(t_learner_scores, "T-learner")
best_s, cv_results["S-learner (tuned)"] = tune(s_learner_scores, "S-learner")

# ---------- 5. Final check on the TEST set (done once) ----------
y_te = test["conversion"].to_numpy()
t_te = test["treatment"].to_numpy()

final_scores = {
    "Heuristic (past spend)": heuristic_scores(None, train, test),
    "T-learner (Step 3 params)": t_learner_scores(baseline_params, train, test),
    "T-learner (tuned)": t_learner_scores(best_t, train, test),
    "S-learner (tuned)": s_learner_scores(best_s, train, test),
}


def bootstrap_qini(y, t, score, n_boot=200):
    """How much does Qini change if we re-draw the test customers?"""
    rng = np.random.default_rng(SEED)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(qini_coefficient(y[idx], t[idx], score[idx]))
    return np.percentile(vals, [5, 95])


print("\nTEST SET RESULTS")
rows = []
for name, s in final_scores.items():
    q = qini_coefficient(y_te, t_te, s)
    lo, hi = bootstrap_qini(y_te, t_te, s)
    model_20, rand_20 = extra_buyers_at(y_te, t_te, s, 0.2)
    model_50, rand_50 = extra_buyers_at(y_te, t_te, s, 0.5)
    rows.append(
        {
            "method": name,
            "test_qini": round(q, 2),
            "range_90pct": f"{lo:.1f} to {hi:.1f}",
            "extra_buyers_top20": round(model_20, 1),
            "random_top20": round(rand_20, 1),
            "extra_buyers_top50": round(model_50, 1),
            "random_top50": round(rand_50, 1),
        }
    )
results = pd.DataFrame(rows)
print(results.to_string(index=False))

# ---------- 6. Chart ----------
plt.figure(figsize=(8, 5))
for name, s in final_scores.items():
    x, q = qini_curve(y_te, t_te, s)
    plt.plot(x, q, label=name)
plt.plot(x, x * q[-1], "k--", label="Random targeting")
plt.xlabel("Fraction of customers targeted")
plt.ylabel("Extra buyers caused by email")
plt.title("Qini curves on the test set")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "step4_comparison.png", dpi=150)

# ---------- 7. Save ----------
# We save the tuned T-learner (the approach we planned to build on).
# Which one to use finally is decided in Step 5 from the numbers above.
m_t = build_model(best_t).fit(
    train.loc[train.treatment == 1, feature_cols],
    train.loc[train.treatment == 1, "conversion"],
)
m_c = build_model(best_t).fit(
    train.loc[train.treatment == 0, feature_cols],
    train.loc[train.treatment == 0, "conversion"],
)
joblib.dump(m_t, MODEL_DIR / "tuned_model_treated.joblib")
joblib.dump(m_c, MODEL_DIR / "tuned_model_control.joblib")

with open(OUT_DIR / "step4_best_params.json", "w") as f:
    json.dump({"t_learner": best_t, "s_learner": best_s}, f, indent=2)

out = test.copy()
for name, s in final_scores.items():
    out["score_" + name.split(" (")[0].lower().replace("-", "_").replace(" ", "_")
        + ("_tuned" if "tuned" in name else "_base" if "Step 3" in name else "")] = s
out.to_csv(PROC_DIR / "test_with_tuned_scores.csv", index=False)

results.to_csv(OUT_DIR / "step4_results.csv", index=False)
print("\nSaved: models, best params, chart, and results in outputs/")