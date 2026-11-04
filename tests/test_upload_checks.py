"""
Checks for the upload cleaning code.
Run from the project root:  python -m pytest -v
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))

from features import build_features  # noqa: E402
from upload_checks import OUTPUT_ORDER, clean_upload  # noqa: E402


def good():
    """Only the required columns."""
    return pd.DataFrame(
        {
            "recency": [10, 6],
            "history": [142.44, 329.08],
            "mens": [1, 1],
            "womens": [0, 1],
            "zip_code": ["Suburban", "Rural"],
            "newbie": [0, 1],
            "channel": ["Phone", "Web"],
        }
    )


def test_good_file_passes_without_history_segment():
    clean, problems, notes = clean_upload(good())
    assert problems == []
    assert notes == []
    assert list(clean.columns) == OUTPUT_ORDER


def test_segment_is_worked_out_from_history():
    clean, _, _ = clean_upload(good())
    assert clean["history_segment"].tolist() == ["2) $100 - $200", "3) $200 - $350"]


def test_provided_segment_is_ignored_even_if_wrong():
    df = good()
    df["history_segment"] = ["nonsense", "also wrong"]
    clean, problems, _ = clean_upload(df)
    assert problems == []
    assert clean["history_segment"].tolist() == ["2) $100 - $200", "3) $200 - $350"]


def test_column_names_ignore_case_and_spaces():
    df = good()
    df.columns = [" Recency ", "HISTORY", "Mens", "Womens", "Zip_Code", "Newbie", "Channel"]
    clean, problems, _ = clean_upload(df)
    assert problems == []
    assert len(clean) == 2


def test_missing_column_is_reported():
    clean, problems, _ = clean_upload(good().drop(columns=["channel"]))
    assert clean is None
    assert "channel" in problems[0]


def test_bad_values_are_reported_with_row_numbers():
    df = good()
    df.loc[1, "channel"] = "Fax"
    df.loc[0, "recency"] = 13
    clean, problems, _ = clean_upload(df)
    assert clean is None
    text = " ".join(problems)
    assert "channel" in text and "recency" in text
    assert "Excel row(s): 3" in text      # DataFrame row 1 is row 3 in Excel (row 1 = header)


def test_money_symbols_are_removed():
    df = good()
    df["history"] = ["$1,142.44", "329.08"]
    clean, problems, _ = clean_upload(df)
    assert problems == []
    assert clean["history"].iloc[0] == 1142.44


def test_yes_no_flags_are_accepted():
    df = good()
    df["mens"] = ["yes", "No"]
    clean, problems, _ = clean_upload(df)
    assert problems == []
    assert clean["mens"].tolist() == [1, 0]


def test_old_spelling_is_fixed():
    df = good()
    df.loc[0, "zip_code"] = "Surburban"
    clean, problems, _ = clean_upload(df)
    assert problems == []
    assert clean["zip_code"].iloc[0] == "Suburban"


def test_empty_file_is_reported():
    clean, problems, _ = clean_upload(good().iloc[0:0])
    assert clean is None
    assert problems


def test_clean_output_feeds_feature_builder():
    with open(ROOT / "data" / "processed" / "feature_cols.json") as f:
        feature_cols = json.load(f)
    clean, _, _ = clean_upload(good())
    X = build_features(clean, feature_cols)
    assert X.shape == (2, len(feature_cols))


def test_worked_out_segment_matches_the_original_dataset():
    """The spending group we compute should match the one in the original data."""
    raw = pd.read_csv(ROOT / "data" / "hillstrom.csv").sample(5000, random_state=0)
    clean, problems, _ = clean_upload(raw)
    assert problems == []
    ours = clean["history_segment"].str[0].to_numpy()
    original = raw["history_segment"].astype(str).str[0].to_numpy()
    assert (ours == original).mean() >= 0.999