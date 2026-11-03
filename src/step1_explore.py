
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

DATA_PATH = Path("data/hillstrom.csv")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)


df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print("\nColumns:", list(df.columns))
print("\nMissing values:\n", df.isnull().sum())


df["treatment"] = (df["segment"] != "No E-Mail").astype(int)

print("\nCustomers per group:")
print(df["segment"].value_counts())
print("\nTreated (1) vs Control (0):")
print(df["treatment"].value_counts())


summary = df.groupby("treatment").agg(
    customers=("conversion", "size"),
    visit_rate=("visit", "mean"),
    conversion_rate=("conversion", "mean"),
    avg_spend=("spend", "mean"),
)
print("\nOutcome by group:\n", summary.round(4))

ate = (
    summary.loc[1, "conversion_rate"] - summary.loc[0, "conversion_rate"]
)
print(f"\nAverage uplift in conversion (email - no email): {ate:.4f}")


treated = df.loc[df["treatment"] == 1, "conversion"]
control = df.loc[df["treatment"] == 0, "conversion"]
t_stat, p_value = stats.ttest_ind(treated, control, equal_var=False)
print(f"p-value: {p_value:.6f}  (below 0.05 means the effect is real)")


print("\nBalance check (mean of features by group):")
numeric_cols = ["recency", "history", "mens", "womens", "newbie"]
print(df.groupby("treatment")[numeric_cols].mean().round(3))

for col in numeric_cols:
    _, p = stats.ttest_ind(
        df.loc[df["treatment"] == 1, col],
        df.loc[df["treatment"] == 0, col],
        equal_var=False,
    )
    flag = "OK" if p > 0.05 else "CHECK"
    print(f"  {col:10s} p={p:.3f}  {flag}")


sns.set_theme(style="whitegrid")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

sns.barplot(data=df, x="segment", y="conversion", ax=axes[0])
axes[0].set_title("Conversion rate by group")

sns.barplot(data=df, x="segment", y="spend", ax=axes[1])
axes[1].set_title("Average spend by group")

sns.barplot(data=df, x="channel", y="conversion", hue="treatment", ax=axes[2])
axes[2].set_title("Conversion by channel (0 = no email, 1 = email)")

plt.tight_layout()
plt.savefig(OUT_DIR / "step1_overview.png", dpi=150)
print("\nSaved chart to outputs/step1_overview.png")


for col in ["channel", "zip_code", "history_segment"]:
    table = (
        df.groupby([col, "treatment"])["conversion"]
        .mean()
        .unstack()
        .rename(columns={0: "control", 1: "email"})
    )
    table["uplift"] = table["email"] - table["control"]
    print(f"\nUplift by {col}:\n", table.round(4))