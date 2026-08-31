import pandas as pd
import matplotlib.pyplot as plt


# Read experiment results
df = pd.read_csv("results/proposed.csv")

# Convert fairness from [0, 1] to percentage
df["fairness_percent"] = (
    df["representation_fairness"] * 100
)

# Average fairness
average_fairness = (
    df["fairness_percent"].mean()
)


plt.figure(figsize=(8, 5))

plt.plot(
    df["round"],
    df["fairness_percent"],
    marker="o",
    linewidth=2,
    label="Representation Fairness"
)

plt.axhline(
    average_fairness,
    linestyle="--",
    label=f"Average = {average_fairness:.1f}%"
)

plt.xlabel("Training Round")
plt.ylabel("Representation Fairness (%)")

plt.title(
    "Data Distribution Representation Fairness Across Training Rounds"
)

plt.ylim(0, 105)

plt.grid(
    True,
    linestyle="--",
    alpha=0.5
)

plt.legend()
plt.tight_layout()

plt.savefig(
    "results/representation_fairness.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()