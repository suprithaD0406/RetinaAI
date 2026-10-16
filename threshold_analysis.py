
from pathlib import Path
import csv

import torch
from torch.utils.data import DataLoader

from dataset import RetinalSegmentationDataset
from model import UNet


# ==================================================
# Configuration
# ==================================================

DATA_DIR = Path("prepared_data")
CHECKPOINT_PATH = Path("checkpoints/best_unet.pth")
OUTPUT_DIR = Path("evaluation_results")
OUTPUT_CSV = OUTPUT_DIR / "threshold_analysis.csv"

THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]
BATCH_SIZE = 1
NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==================================================
# Metrics
# ==================================================

@torch.no_grad()
def calculate_counts(probabilities, masks, threshold):
    predictions = probabilities >= threshold
    targets = masks >= 0.5

    tp = (predictions & targets).sum().item()
    fp = (predictions & ~targets).sum().item()
    fn = (~predictions & targets).sum().item()
    tn = (~predictions & ~targets).sum().item()

    return tp, fp, fn, tn


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


# ==================================================
# Main
# ==================================================

def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH.resolve()}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Device:", DEVICE)
    print("Checkpoint:", CHECKPOINT_PATH.resolve())
    print("Dataset split: validation")
    print("Thresholds:", THRESHOLDS)

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
        split="validation",
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    totals = {
        threshold: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for threshold in THRESHOLDS
    }

    image_metrics = {
        threshold: {"dice_sum": 0.0, "iou_sum": 0.0}
        for threshold in THRESHOLDS
    }

    num_images = 0

    with torch.inference_mode():
        for images, masks in loader:
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=(DEVICE.type == "cuda"),
            ):
                logits = model(images)

            probabilities = torch.sigmoid(logits.float())
            targets = masks >= 0.5
            num_images += images.size(0)

            for threshold in THRESHOLDS:
                tp, fp, fn, tn = calculate_counts(
                    probabilities, masks, threshold
                )

                totals[threshold]["tp"] += tp
                totals[threshold]["fp"] += fp
                totals[threshold]["fn"] += fn
                totals[threshold]["tn"] += tn

                predictions = probabilities >= threshold

                # Per-image Dice and IoU.
                pred_flat = predictions.flatten(start_dim=1)
                target_flat = targets.flatten(start_dim=1)

                intersection = (
                    pred_flat & target_flat
                ).sum(dim=1).float()

                pred_count = pred_flat.sum(dim=1).float()
                target_count = target_flat.sum(dim=1).float()
                union = (pred_flat | target_flat).sum(dim=1).float()

                dice = (
                    2 * intersection + 1e-7
                ) / (
                    pred_count + target_count + 1e-7
                )

                iou = (
                    intersection + 1e-7
                ) / (union + 1e-7)

                image_metrics[threshold]["dice_sum"] += dice.sum().item()
                image_metrics[threshold]["iou_sum"] += iou.sum().item()

    rows = []

    for threshold in THRESHOLDS:
        counts = totals[threshold]

        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        tn = counts["tn"]

        dice = safe_divide(2 * tp, 2 * tp + fp + fn)
        iou = safe_divide(tp, tp + fp + fn)
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        accuracy = safe_divide(tp + tn, tp + tn + fp + fn)

        macro_dice = safe_divide(
            image_metrics[threshold]["dice_sum"], num_images
        )
        macro_iou = safe_divide(
            image_metrics[threshold]["iou_sum"], num_images
        )

        rows.append({
            "threshold": threshold,
            "global_dice": dice,
            "global_iou": iou,
            "mean_image_dice": macro_dice,
            "mean_image_iou": macro_iou,
            "precision": precision,
            "recall": recall,
            "pixel_accuracy": accuracy,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
        })

    columns = list(rows[0].keys())

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nValidation images evaluated: {num_images}")
    print("\nVALIDATION THRESHOLD COMPARISON")
    print("-" * 100)
    print(
        f"{'Threshold':>9} {'Global Dice':>12} {'Mean Dice':>11} "
        f"{'Global IoU':>11} {'Precision':>11} {'Recall':>10}"
    )

    for row in rows:
        print(
            f"{row['threshold']:>9.2f} "
            f"{row['global_dice']:>12.4f} "
            f"{row['mean_image_dice']:>11.4f} "
            f"{row['global_iou']:>11.4f} "
            f"{row['precision']:>11.4f} "
            f"{row['recall']:>10.4f}"
        )

    print("\nSaved comparison to:", OUTPUT_CSV.resolve())
    print(
        "\nUse validation results to select a threshold. "
        "Do not use the official test set to tune it."
    )


if __name__ == "__main__":
    main()
