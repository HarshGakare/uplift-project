"""Scoring logic shared by the API. Must match Step 2 feature building."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


class UpliftService:
    def __init__(self, root: Path):
        self.policy = json.loads((root / "outputs" / "policy.json").read_text())
        self.features = json.loads(
            (root / "data" / "processed" / "features.json").read_text()
        )
        self.method = self.policy["method"]
        self.cutoff = self.policy.get("score_cutoff")

        self.models = {}
        if self.method == "T-learner (tuned)":
            self.models["t"] = joblib.load(root / "models" / "model_t_tuned.joblib")
            self.models["c"] = joblib.load(root / "models" / "model_c_tuned.joblib")
        elif self.method == "T-learner (Step 3, untuned)":
            self.models["t"] = joblib.load(root / "models" / "model_t.joblib")
            self.models["c"] = joblib.load(root / "models" / "model_c.joblib")
        else:
            self.models["transformed"] = joblib.load(
                root / "models" / "model_transformed.joblib"
            )

        self.zip_options = [
            f[len("zip_code_"):] for f in self.features if f.startswith("zip_code_")
        ]
        self.channel_options = [
            f[len("channel_"):] for f in self.features if f.startswith("channel_")
        ]

    def validate_categories(self, raw: pd.DataFrame) -> None:
        bad_zip = sorted(set(raw["zip_code"]) - set(self.zip_options))
        bad_channel = sorted(set(raw["channel"]) - set(self.channel_options))
        problems = []
        if bad_zip:
            problems.append(f"Unknown zip_code {bad_zip}. Allowed: {self.zip_options}")
        if bad_channel:
            problems.append(
                f"Unknown channel {bad_channel}. Allowed: {self.channel_options}"
            )
        if problems:
            raise ValueError(" | ".join(problems))

    def build_features(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.copy()
        df["log_history"] = np.log1p(df["history"])
        df["is_recent"] = (df["recency"] <= 3).astype(int)
        df["both_products"] = ((df["mens"] == 1) & (df["womens"] == 1)).astype(int)

        out = pd.DataFrame(index=df.index)
        for feat in self.features:
            if feat.startswith("zip_code_"):
                out[feat] = (df["zip_code"] == feat[len("zip_code_"):]).astype(int)
            elif feat.startswith("channel_"):
                out[feat] = (df["channel"] == feat[len("channel_"):]).astype(int)
            else:
                out[feat] = df[feat]
        return out[self.features]

    def score(self, raw: pd.DataFrame) -> pd.DataFrame:
        feats = self.build_features(raw)
        result = pd.DataFrame(index=raw.index)
        if "transformed" in self.models:
            result["uplift"] = self.models["transformed"].predict(feats)
        else:
            result["p_email"] = self.models["t"].predict_proba(feats)[:, 1]
            result["p_no_email"] = self.models["c"].predict_proba(feats)[:, 1]
            result["uplift"] = result["p_email"] - result["p_no_email"]
        return result

    def decide(self, uplift_value: float) -> bool:
        """True = email, False = skip."""
        return self.cutoff is not None and uplift_value >= self.cutoff