
from pathlib import Path
import csv

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from dataset import RetinalSegmentationDataset
from model import UNet


# ==================================================
# Configuration
# ==================================================

DATA_DIR = Path("prepared_data")
CHECKPOINT_PATH = Path("checkpoints/best_unet.pth")

# Keep previous evaluation results untouched.
OUTPUT_DIR = Path("evaluation_results_threshold_070")
OUTPUT_CSV = OUTPUT_DIR / "test_metrics.csv"

THRESHOLD = 0.70
BATCH_SIZE = 1
NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==================================================
# Metrics
# ==================================================

def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def calculate_counts(prediction, target):
    prediction = prediction.bool()
    target = target.bool()

    tp = (prediction & target).sum().item()
    fp = (prediction & ~target).sum().item()
    fn = (~prediction & target).sum().item()
    tn = (~prediction & ~target).sum().item()

    return tp, fp, fn, tn


def create_visualization(image, target, prediction, output_path):
    image = image.permute(1, 2, 0).cpu().numpy()
    target = target.squeeze().cpu().numpy().astype(bool)
    prediction = prediction.squeeze().cpu().numpy().astype(bool)

    # Mark predicted pixels in red on the original image.
    overlay = image.copy()
    overlay[prediction] = (
        0.55 * overlay[prediction]
        + 0.45 * np.array([1.0, 0.0, 0.0])
    )

    fig, axes = plt.subplots(1, 4, figsize=(12, 4))

    axes[0].imshow(image)
    axes[0].set_title("Retinal image")

    axes[1].imshow(target, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground truth")

    axes[2].imshow(prediction, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"Prediction ({THRESHOLD:.2f})")

    axes[3].imshow(overlay)
    axes[3].set_title("Prediction overlay")

    for axis in axes:
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ==================================================
# Main evaluation
# ==================================================

def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH.resolve()}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Device:", DEVICE)
    print("Checkpoint:", CHECKPOINT_PATH.resolve())
    print("Threshold:", THRESHOLD)
    print("Dataset split: official test set")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model = UNet(
        in_channels=3,
        out_channels=1,
        base_channels=checkpoint.get("base_channels", 32),
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = RetinalSegmentationDataset(
        root_dir=str(DATA_DIR),
        split="test",
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    tp_total = 0
    fp_total = 0
    fn_total = 0
    tn_total = 0

    dice_sum = 0.0
    iou_sum = 0.0
    image_count = 0

    image_rows = []

    with torch.inference_mode():
        for index, (images, masks) in enumerate(loader, start=1):
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=(DEVICE.type == "cuda"),
            ):
                logits = model(images)

            probabilities = torch.sigmoid(logits.float())
            predictions = probabilities >= THRESHOLD
            targets = masks >= 0.5

            tp, fp, fn, tn = calculate_counts(
                predictions, targets
            )

            tp_total += tp
            fp_total += fp
            fn_total += fn
            tn_total += tn

            intersection = (predictions & targets).sum().item()
            pred_count = predictions.sum().item()
            target_count = targets.sum().item()
            union = (predictions | targets).sum().item()

            dice = safe_divide(
                2 * intersection, pred_count + target_count
            )
            iou = safe_divide(intersection, union)

            dice_sum += dice
            iou_sum += iou
            image_count += 1

            image_name = f"test_{index:02d}.png"

            create_visualization(
                images[0],
                masks[0],
                predictions[0],
                OUTPUT_DIR / image_name,
            )

            image_rows.append({
                "image": image_name,
                "dice": dice,
                "iou": iou,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
            })

    global_dice = safe_divide(
        2 * tp_total, 2 * tp_total + fp_total + fn_total
    )
    global_iou = safe_divide(
        tp_total, tp_total + fp_total + fn_total
    )
    precision = safe_divide(tp_total, tp_total + fp_total)
    recall = safe_divide(tp_total, tp_total + fn_total)
    accuracy = safe_divide(
        tp_total + tn_total,
        tp_total + tn_total + fp_total + fn_total,
    )

    mean_dice = safe_divide(dice_sum, image_count)
    mean_iou = safe_divide(iou_sum, image_count)

    summary = {
        "threshold": THRESHOLD,
        "test_images": image_count,
        "global_dice": global_dice,
        "global_iou": global_iou,
        "mean_image_dice": mean_dice,
        "mean_image_iou": mean_iou,
        "pixel_precision": precision,
        "pixel_recall": recall,
        "pixel_accuracy": accuracy,
        "true_positives": tp_total,
        "false_positives": fp_total,
        "false_negatives": fn_total,
        "true_negatives": tn_total,
    }

    with OUTPUT_CSV.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(summary.keys())
        )
        writer.writeheader()
        writer.writerow(summary)

    with (OUTPUT_DIR / "per_image_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(image_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(image_rows)

    print("\nFINAL TEST RESULTS")
    print("=" * 45)
    print("Test images:", image_count)
    print(f"Threshold: {THRESHOLD:.2f}")
    print(f"Global Dice: {global_dice:.4f}")
    print(f"Global IoU: {global_iou:.4f}")
    print(f"Mean image Dice: {mean_dice:.4f}")
    print(f"Mean image IoU: {mean_iou:.4f}")
    print(f"Pixel precision: {precision:.4f}")
    print(f"Pixel recall: {recall:.4f}")
    print(f"Pixel accuracy: {accuracy:.4f}")

    print("\nConfusion counts")
    print("TP:", tp_total)
    print("FP:", fp_total)
    print("FN:", fn_total)
    print("TN:", tn_total)

    print("\nSaved results to:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
