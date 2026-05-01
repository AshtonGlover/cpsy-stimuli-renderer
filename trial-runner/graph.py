import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

IN_PATH = "data/raw data/erv_labels.csv"
OUT_PATH = "data/distances/erv_distances.csv"
PARTIC_ID = "ERV"

# =========================
# 1. LOAD
# =========================
df = pd.read_csv(IN_PATH)
df["response"] = df["response"].str.lower()

# =========================
# 2. EXTRACT FRAME
# =========================
df["frame"] = df["image_name"].str.extract(r"(\d+)").astype(int)

# =========================
# 3. INVERSE MAP (FOR ROTATION)
# =========================
inverse_map = {
    36: 84,
    42: 78,
    48: 72,
    54: 66,
    60: 60,
    66: 54,
    72: 48,
    78: 42,
    84: 36
}

def canonical_frame(row):
    f = row["frame"]
    if row["rotation_degrees"] == 180:
        return inverse_map[f]
    return f

df["frame_canonical"] = df.apply(canonical_frame, axis=1)

# =========================
# 4. DEFINE ACCURACY
# =========================
# 0°: top correct
# 180°: bottom correct
df["correct"] = (
    ((df["rotation_degrees"] == 0) & (df["response"] == "top")) |
    ((df["rotation_degrees"] == 180) & (df["response"] == "bottom"))
).astype(int)

# =========================
# 5. AGGREGATE
# =========================
summary = (
    df.groupby(["category", "frame_canonical"])
      .agg(
          accuracy=("correct", "mean"),
          n=("correct", "count")
      )
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
# 7. PLOT (SMOOTH CURVES)
# =========================
plt.figure(figsize=(8, 5))

x_smooth = np.linspace(min(order), max(order), 300)
categories = ["shading", "shadow", "incongruent"]

for cat in categories:
    sub = summary[summary["category"] == cat].sort_values("frame_canonical")

    x = sub["frame_canonical"].astype(float).values
    y = sub["accuracy"].values

    # remove any bad values just in case
    mask = ~np.isnan(y)
    x = x[mask]
    y = y[mask]

    if cat == "shadow":
        x = 78 - (x - 42)  # reflects around midpoint of [42,84]

    # plot points + straight connecting lines
    plt.plot(
        x,
        y,
        marker="o",
        linestyle="-",
        label=cat
    )

plt.axhline(0.5, color="black", linewidth=1)

plt.xlabel("Light Direction")
plt.ylabel("Accuracy")
plt.title("Convexity Perception Accuracy Across Light Direction (Participant " + PARTIC_ID +" )")
plt.xticks([42, 78], ["Light from above", "Light from below"])
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()

plt.show()

# =========================
# 8. CURVE DISTANCE ANALYSIS
# =========================

summary = (
    summary.groupby(["frame_canonical", "category"], as_index=False)
           .agg(accuracy=("accuracy", "mean"))
)

pivot = summary.pivot(
    index="frame_canonical",
    columns="category",
    values="accuracy"
).sort_index()

# extract curves
inc = pivot["incongruent"].values
sha = pivot["shading"].values
sho = pivot["shadow"].values

# squared distances per frame
dist_shading = (inc - sha) ** 2
dist_shadow = (inc - sho) ** 2

# build output table
dist_df = pd.DataFrame({
    "frame": pivot.index,
    "dist_shading": dist_shading,
    "dist_shadow": dist_shadow
})

# save next to original CSV
out_path = OUT_PATH
dist_df.to_csv(out_path, index=False)

print("\nSaved curve distances to:", out_path)
print("\nTOTAL DISTANCES")
print("Shading:", np.mean(dist_shading))
print("Shadow:", np.mean(dist_shadow))