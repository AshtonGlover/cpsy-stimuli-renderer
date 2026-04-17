import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

inverse_map = {
    42: 78,
    48: 72,
    54: 66,
    60: 60,
    66: 54,
    72: 48,
    78: 42
}

# =========================
# 1. LOAD
# =========================
df = pd.read_csv("results/ep/ep_labels.csv")
df["response"] = df["response"].str.lower()

# =========================
# 2. KEEP ONLY UNROTATED
# =========================
df = df[df["rotation_degrees"] == 0].copy()

# =========================
# 3. EXTRACT FRAME
# =========================
df["frame"] = df["image_name"].str.extract(r"(\d+)").astype(int)

def collapse(row):
    f = row["frame"]
    if row["rotation_degrees"] == 180:
        return inverse_map[f]
    return f

df["frame_canonical"] = df.apply(collapse, axis=1)

# =========================
# 4. ENCODE PERCEPT (SIGNED)
# =========================
df["percept"] = df["response"].map({
    "top": 1,
    "bottom": -1
})


# =========================
# 5. AGGREGATE (PSYCHOMETRIC SIGNAL)
# =========================
summary = (
    df.groupby(["category", "frame_canonical"])
      .agg(mean_percept=("percept", "mean"))
      .reset_index()
)

# =========================
# 6. ORDER FRAMES
# =========================
order = [42, 48, 54, 60, 66, 72, 78]

summary["frame_canonical"] = pd.Categorical(
    summary["frame_canonical"],
    categories=order,
    ordered=True
)

# =========================
# 7. PLOT PSYCHOMETRIC CURVES
# =========================
plt.figure(figsize=(8, 5))

categories = ["shading", "shadow", "incongruent"]

for cat in categories:
    sub = summary[summary["category"] == cat].sort_values("frame_canonical")
    
    plt.plot(
        sub["frame_canonical"],
        sub["mean_percept"],
        marker="o",
        label=cat
    )

plt.axhline(0, color="black", linewidth=1)

plt.xlabel("Light Direction Frame")
plt.ylabel("Perceptual Bias (Top ↔ Bottom Convex)")
plt.title("Psychometric Function Across Light Direction (Unrotated Only)")
plt.xticks(order)
plt.ylim(-1, 1)
plt.legend()
plt.tight_layout()

plt.show()