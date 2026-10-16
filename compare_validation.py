import os
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import RetinalSegmentationDataset
from model import UNet
from model_pretrained import PretrainedUNet

DATA_DIR = "prepared_data"
OUTPUT_DIR = "validation_model_comparison"
THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def normalize_pretrained(images):
    return (images - MEAN.to(images.device)) / STD.to(images.device)


def load_baseline():
    model = UNet(base_channels=32).to(DEVICE)
    checkpoint = torch.load(
        os.path.join("checkpoints", "best_unet.pth"),
        map_location=DEVICE,
        weights_only=False,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    else:
        state = checkpoint

    model.load_state_dict(state)
    model.eval()
    return model


def load_pretrained():
    model = PretrainedUNet(pretrained=False).to(DEVICE)
    checkpoint = torch.load(
        os.path.join(
            "checkpoints_pretrained", "best_pretrained_unet.pth"
        ),
        map_location=DEVICE,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def metrics(logits, masks):
    pred = (torch.sigmoid(logits) >= THRESHOLD).float()
    target = (masks >= 0.5).float()

    dims = (1, 2, 3)
    tp = (pred * target).sum(dim=dims)
    fp = (pred * (1 - target)).sum(dim=dims)
    fn = ((1 - pred) * target).sum(dim=dims)

    dice = (2 * tp + 1e-7) / (2 * tp + fp + fn + 1e-7)
    iou = (tp + 1e-7) / (tp + fp + fn + 1e-7)
    precision = (tp + 1e-7) / (tp + fp + 1e-7)
    recall = (tp + 1e-7) / (tp + fn + 1e-7)

    return {
        "dice": dice.item(),
        "iou": iou.item(),
        "precision": precision.item(),
        "recall": recall.item(),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dataset = RetinalSegmentationDataset(DATA_DIR, "validation")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    baseline = load_baseline()
    pretrained = load_pretrained()

    rows = []
    summary = {
        "baseline": {"dice": [], "iou": [], "precision": [], "recall": []},
        "pretrained": {"dice": [], "iou": [], "precision": [], "recall": []},
    }

    print("Device:", DEVICE)
    print("Validation images:", len(dataset))
    print("Threshold:", THRESHOLD)

    with torch.inference_mode():
        for index, (images, masks) in enumerate(loader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            baseline_logits = baseline(images)
            pretrained_logits = pretrained(normalize_pretrained(images))

            baseline_metrics = metrics(baseline_logits, masks)
            pretrained_metrics = metrics(pretrained_logits, masks)

            row = [index + 1]
            for model_name, result in (
                ("baseline", baseline_metrics),
                ("pretrained", pretrained_metrics),
            ):
                for metric_name in ("dice", "iou", "precision", "recall"):
                    value = result[metric_name]
                    summary[model_name][metric_name].append(value)
                    row.append(value)

            rows.append(row)

            # Show image, ground truth, and both predictions.
            original = images[0].cpu().permute(1, 2, 0).numpy()
            target = masks[0, 0].cpu().numpy() >= 0.5
            base_pred = (
                torch.sigmoid(baseline_logits[0, 0]).cpu().numpy()
                >= THRESHOLD
            )
            pre_pred = (
                torch.sigmoid(pretrained_logits[0, 0]).cpu().numpy()
                >= THRESHOLD
            )

            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            axes[0].imshow(np.clip(original, 0, 1))
            axes[0].set_title("Retinal image")
            axes[1].imshow(target, cmap="gray", vmin=0, vmax=1)
            axes[1].set_title("Ground truth")
            axes[2].imshow(base_pred, cmap="gray", vmin=0, vmax=1)
            axes[2].set_title(
                f"Original U-Net\nDice {baseline_metrics['dice']:.3f}"
            )
            axes[3].imshow(pre_pred, cmap="gray", vmin=0, vmax=1)
            axes[3].set_title(
                f"Pretrained U-Net\nDice {pretrained_metrics['dice']:.3f}"
            )

            for ax in axes:
                ax.axis("off")

            fig.tight_layout()
            fig.savefig(
                os.path.join(OUTPUT_DIR, f"comparison_{index + 1:02d}.png"),
                dpi=120,
                bbox_inches="tight",
            )
            plt.close(fig)

    csv_path = os.path.join(OUTPUT_DIR, "per_image_metrics.csv")
    with open(csv_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "image",
            "baseline_dice", "baseline_iou",
            "baseline_precision", "baseline_recall",
            "pretrained_dice", "pretrained_iou",
            "pretrained_precision", "pretrained_recall",
        ])
        writer.writerows(rows)

    print("\n--- Mean per-image validation metrics ---")
    for name in ("baseline", "pretrained"):
        print(f"\n{name.upper()}")
        for metric_name, values in summary[name].items():
            print(f"{metric_name}: {np.mean(values):.4f}")

    print("\nSaved comparisons:", OUTPUT_DIR)
    print("Saved metrics:", csv_path)


if __name__ == "__main__":
    main()