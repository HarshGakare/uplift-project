"""Friendly checks and cleaning for an uploaded customer file."""
import numpy as np
import pandas as pd

MAX_ROWS = 100_000

# Columns the customer must provide.
# The spending group ("history_segment") is NOT here: we work it out from "history".
REQUIRED = ["recency", "history", "mens", "womens", "zip_code", "newbie", "channel"]
OUTPUT_ORDER = [
    "recency", "history_segment", "history", "mens", "womens", "zip_code", "newbie", "channel",
]

BAND_LABELS = [
    "1) $0 - $100", "2) $100 - $200", "3) $200 - $350", "4) $350 - $500",
    "5) $500 - $750", "6) $750 - $1,000", "7) $1,000 +",
]
BAND_EDGES = [-np.inf, 100, 200, 350, 500, 750, 1000, np.inf]

ZIP_MAP = {"rural": "Rural", "suburban": "Suburban", "surburban": "Suburban", "urban": "Urban"}
CHANNEL_MAP = {"phone": "Phone", "web": "Web", "multichannel": "Multichannel"}
YES = {"1", "1.0", "yes", "y", "true", "t"}
NO = {"0", "0.0", "no", "n", "false", "f"}


def _rows(bad):
    """Excel-style row numbers (the header is row 1) of the first few bad rows."""
    numbers = [int(i) + 2 for i in bad[bad].index[:5]]
    text = ", ".join(str(n) for n in numbers)
    return text + (" ..." if int(bad.sum()) > 5 else "")


def _message(column, bad, expected):
    return (
        f"Column '{column}': {int(bad.sum())} row(s) have a missing or invalid value. "
        f"Expected {expected}. Check Excel row(s): {_rows(bad)}."
    )


def _flag(series):
    """Turns 1/0, yes/no, true/false into 1 or 0 (anything else becomes NaN)."""
    text = series.astype(str).str.strip().str.lower()
    out = pd.Series(np.nan, index=series.index)
    out[text.isin(YES)] = 1
    out[text.isin(NO)] = 0
    return out


def clean_upload(raw):
    """
    Returns (clean_table, problems, notes).
    If problems is not empty, clean_table is None and problems explains what to fix.
    Extra columns in the file (and a 'history_segment' column, if present) are ignored.
    """
    problems, notes = [], []

    if raw is None or len(raw) == 0:
        return None, ["The file has no customer rows."], notes
    if len(raw) > MAX_ROWS:
        return None, [f"The file has {len(raw):,} rows. Please upload at most {MAX_ROWS:,} customers at a time."], notes

    df = raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        return None, [
            "These columns are missing: " + ", ".join(missing) + ". Check the spelling of the "
            "column names in the first row of your file. If all your data shows up in one "
            "column, save the file again as 'CSV (Comma delimited)'."
        ], notes

    out = pd.DataFrame(index=df.index)

    # recency: whole number 1 to 12
    recency = pd.to_numeric(df["recency"], errors="coerce")
    bad = recency.isna() | (recency < 1) | (recency > 12) | (recency != recency.round())
    if bad.any():
        problems.append(_message("recency", bad, "a whole number from 1 to 12"))
    out["recency"] = recency

    # history: money, a $ sign or commas are removed
    text = df["history"].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False).str.strip()
    history = pd.to_numeric(text, errors="coerce")
    bad = history.isna() | (history < 0)
    if bad.any():
        problems.append(_message("history", bad, "a number of dollars, 0 or more (for example 142.44)"))
    out["history"] = history

    # 1/0 columns
    for col in ("mens", "womens", "newbie"):
        flag = _flag(df[col])
        bad = flag.isna()
        if bad.any():
            problems.append(_message(col, bad, "1 for yes or 0 for no"))
        out[col] = flag

    # area type
    zip_code = df["zip_code"].astype(str).str.strip().str.lower().map(ZIP_MAP)
    bad = zip_code.isna()
    if bad.any():
        problems.append(_message("zip_code", bad, "Rural, Suburban or Urban"))
    out["zip_code"] = zip_code

    # buying channel
    channel = df["channel"].astype(str).str.strip().str.lower().str.replace(" ", "", regex=False).map(CHANNEL_MAP)
    bad = channel.isna()
    if bad.any():
        problems.append(_message("channel", bad, "Phone, Web or Multichannel"))
    out["channel"] = channel

    if problems:
        return None, problems, notes

    # Spending group is always worked out from the dollar amount
    out["history_segment"] = pd.cut(
        history, bins=BAND_EDGES, labels=BAND_LABELS, right=False
    ).astype(object)

    out["recency"] = out["recency"].astype(int)
    for col in ("mens", "womens", "newbie"):
        out[col] = out[col].astype(int)
    return out[OUTPUT_ORDER], [], notes