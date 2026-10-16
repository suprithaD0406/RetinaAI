import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTPUT_DIR = Path("experiment_comparison")
OUTPUT_DIR.mkdir(exist_ok=True)

# Results collected from your completed experiments.
# Validation metrics are kept separate from test metrics.
experiments = [
    {
        "model": "Original U-Net",
        "best_validation_dice": 0.5503,
        "validation_mean_dice": 0.5505,
        "validation_mean_iou": 0.4016,
        "test_global_dice": 0.5258,
        "test_global_iou": 0.3567,
        "test_precision": 0.6037,
        "test_recall": 0.4657,
        "status": "Test evaluated",
    },
    {
        "model": "Tversky U-Net",
        "best_validation_dice": 0.3788,
        "validation_mean_dice": None,
        "validation_mean_iou": None,
        "test_global_dice": None,
        "test_global_iou": None,
        "test_precision": None,
        "test_recall": None,
        "status": "Validation only",
    },
    {
        "model": "Pretrained EfficientNet-B0 U-Net",
        "best_validation_dice": 0.5439,
        "validation_mean_dice": 0.5442,
        "validation_mean_iou": 0.3906,
        "test_global_dice": None,
        "test_global_iou": None,
        "test_precision": None,
        "test_recall": None,
        "status": "Validation only",
    },
]

csv_path = OUTPUT_DIR / "experiment_results.csv"

with csv_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=experiments[0].keys())
    writer.writeheader()
    writer.writerows(experiments)

# Plot validation Dice only, so the comparison uses the same split.
names = [item["model"] for item in experiments]
scores = [item["best_validation_dice"] for item in experiments]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(names, scores)
ax.set_title("RetinaAI: Best Validation Dice by Experiment")
ax.set_ylabel("Best validation Dice")
ax.set_ylim(0, 1)
ax.tick_params(axis="x", rotation=15)

for bar, score in zip(bars, scores):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f"{score:.4f}",
        ha="center",
    )

fig.tight_layout()
plot_path = OUTPUT_DIR / "validation_dice_comparison.png"
fig.savefig(plot_path, dpi=160)
plt.close(fig)

print("Experiment report created.")
print("CSV:", csv_path)
print("Validation chart:", plot_path)
print("\nBest validation Dice:")
for item in experiments:
    print(f"{item['model']}: {item['best_validation_dice']:.4f}")

print("\nNote: Missing metrics are intentionally left blank.")
print("Do not compare validation scores directly with test scores.")