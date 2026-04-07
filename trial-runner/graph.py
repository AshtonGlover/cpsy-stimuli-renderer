from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# =========================
# CONFIG
# =========================
CSV_PATH = Path("results/ep/ep_labels.csv")  # <-- change this
DROP_PRACTICE = True


# =========================
# HELPERS
# =========================
def extract_frame(image_name: str) -> int | None:
    match = re.search(r"(\d+)", image_name)
    return int(match.group(1)) if match else None


def normalize_frame(frame: int, rotation: int, max_frame: int) -> int:
    if rotation == 180:
        return max_frame - frame
    return frame


def logistic(x: np.ndarray, mu: float, s: float) -> np.ndarray:
    return 1 / (1 + np.exp(-(x - mu) / s))


# =========================
# LOAD + PREPROCESS
# =========================
df = pd.read_csv(CSV_PATH)

# extract frame
df["frame"] = df["image_name"].apply(extract_frame)

# drop bad rows early
df = df.dropna(subset=["frame"])

# remove practice trials if desired
if DROP_PRACTICE:
    df = df[df["category"] != "practice"]

# compute max frame for normalization
MAX_FRAME = df["frame"].max()

# normalize lighting direction
df["norm_frame"] = df.apply(
    lambda row: normalize_frame(row["frame"], row["rotation_degrees"], MAX_FRAME),
    axis=1,
)

# response variable: P(top chosen)
df["top_choice"] = (df["response"] == "top").astype(int)


# =========================
# AGGREGATE
# =========================
grouped = (
    df.groupby(["category", "norm_frame"])
    .agg(
        p_top=("top_choice", "mean"),
        n=("top_choice", "count"),
    )
    .reset_index()
)

# binomial standard error
grouped["sem"] = np.sqrt(
    grouped["p_top"] * (1 - grouped["p_top"]) / grouped["n"]
)


# =========================
# PLOTTING + FITTING
# =========================
plt.figure()

x_fit = np.linspace(grouped["norm_frame"].min(), grouped["norm_frame"].max(), 300)

categories = sorted(grouped["category"].unique())

for category in categories:
    subset = grouped[grouped["category"] == category]

    x = subset["norm_frame"].values
    y = subset["p_top"].values
    yerr = subset["sem"].values

    # plot empirical data
    plt.errorbar(x, y, yerr=yerr, fmt="o", label=f"{category} data")

    # fit logistic curve
    if len(x) >= 3:
        try:
            p0 = [np.median(x), 5]  # initial guess
            params, _ = curve_fit(logistic, x, y, p0=p0, maxfev=10000)

            y_fit = logistic(x_fit, *params)
            plt.plot(x_fit, y_fit, label=f"{category} fit")

            print(
                f"{category}: "
                f"bias (mu)={params[0]:.2f}, "
                f"slope={params[1]:.2f}"
            )

        except RuntimeError:
            print(f"Fit failed for {category}")


# =========================
# FINALIZE PLOT
# =========================
plt.xlabel("Normalized Frame (Light Direction)")
plt.ylabel("P(Top Chosen)")
plt.title("Psychometric Curves by Condition")
plt.ylim(0, 1)
plt.legend()

plt.tight_layout()
plt.show()