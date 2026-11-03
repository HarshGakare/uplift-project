"""
Writes requirements files with the exact versions installed on your machine.
Run from the project root:  python src/print_versions.py
"""
import sys
from importlib.metadata import version

app_pkgs = ["streamlit", "pandas", "numpy", "scikit-learn", "lightgbm", "joblib", "matplotlib"]
dev_only = ["optuna", "seaborn", "scipy"]


def lines(pkgs):
    return [f"{p}=={version(p)}" for p in pkgs]


# utf-8 on purpose: PowerShell's ">" redirect can create a file pip cannot read
with open("requirements.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines(app_pkgs)) + "\n")

with open("requirements-dev.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines(app_pkgs + dev_only)) + "\n")

print("Your Python version:", sys.version.split()[0])
print("\nrequirements.txt (used by the live app):")
print(open("requirements.txt", encoding="utf-8").read())
print("Also wrote requirements-dev.txt (full list for running every step).")