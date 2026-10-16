import os
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import RetinalSegmentationDataset
from model_pretrained import PretrainedUNet

DATA_DIR = "prepared_data"
CHECKPOINT_PATH = os.path.join(
    "checkpoints_pretrained", "best_pretrained_unet.pth"
)
OUTPUT_DIR = "evaluation_pretrained_validation"
BATCH_SIZE = 1
THRESHOLD = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def normalize_image(image):
    return (image - MEAN.to(image.device)) / STD.to(image.device)


def calculate_metrics(prediction, target):
    prediction = prediction.astype(bool)
    target = target.astype(bool)

    tp = np.logical_and(prediction, target).sum()
    fp = np.logical_and(prediction, ~target).sum()
    fn = np.logical_and(~prediction, target).sum()
    tn = np.logical_and(~prediction, ~target).sum()

    dice = (2 * tp) / (2 * tp + fp + fn + 1e-7)
    iou = tp / (tp + fp + fn + 1e-7)
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)

    return dice, iou, precision, recall, tp, fp, fn, tn


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dataset = RetinalSegmentationDataset(DATA_DIR, "validation")
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    model = PretrainedUNet(pretrained=False).to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT_PATH, map_location=DEVICE, weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_tp = all_fp = all_fn = all_tn = 0
    per_image_dice = []
    per_image_iou = []
    per_image_precision = []
    per_image_recall = []
    rows = []

    print("Device:", DEVICE)
    print("Split: validation")
    print("Images:", len(dataset))
    print("Threshold:", THRESHOLD)

    with torch.inference_mode():
        for index, (images, masks) in enumerate(loader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            images = normalize_image(images)
            logits = model(images)
            probabilities = torch.sigmoid(logits)

            prediction = (
                probabilities[0, 0].cpu().numpy() >= THRESHOLD
            )
            target = masks[0, 0].cpu().numpy() >= 0.5

            metrics = calculate_metrics(prediction, target)
            dice, iou, precision, recall, tp, fp, fn, tn = metrics

            per_image_dice.append(dice)
            per_image_iou.append(iou)
            per_image_precision.append(precision)
            per_image_recall.append(recall)

            all_tp += int(tp)
            all_fp += int(fp)
            all_fn += int(fn)
            all_tn += int(tn)

            rows.append([
                index + 1, dice, iou, precision, recall,
                int(tp), int(fp), int(fn), int(tn)
            ])

            # Original image for visualization: undo normalization.
            display_image = (
                images[0].cpu() * STD[0] + MEAN[0]
            ).permute(1, 2, 0).numpy()
            display_image = np.clip(display_image, 0, 1)

            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            axes[0].imshow(display_image)
            axes[0].set_title("Retinal image")
            axes[1].imshow(target, cmap="gray", vmin=0, vmax=1)
            axes[1].set_title("Ground truth")
            axes[2].imshow(prediction, cmap="gray", vmin=0, vmax=1)
            axes[2].set_title("Prediction")
            axes[3].imshow(display_image)
            axes[3].imshow(
                prediction, cmap="Reds", alpha=0.45, vmin=0, vmax=1
            )
            axes[3].set_title("Prediction overlay")

            for axis in axes:
                axis.axis("off")

            fig.suptitle(
                f"Validation image {index + 1} | Dice: {dice:.3f}"
            )
            fig.tight_layout()
            fig.savefig(
                os.path.join(
                    OUTPUT_DIR, f"validation_{index + 1:02d}.png"
                ),
                dpi=120,
                bbox_inches="tight",
            )
            plt.close(fig)

    global_dice = (
        2 * all_tp / (2 * all_tp + all_fp + all_fn + 1e-7)
    )
    global_iou = (
        all_tp / (all_tp + all_fp + all_fn + 1e-7)
    )
    global_precision = all_tp / (all_tp + all_fp + 1e-7)
    global_recall = all_tp / (all_tp + all_fn + 1e-7)

    csv_path = os.path.join(OUTPUT_DIR, "validation_metrics.csv")
    with open(csv_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "image", "dice", "iou", "precision", "recall",
            "tp", "fp", "fn", "tn"
        ])
        writer.writerows(rows)

    print("\n--- Validation Results ---")
    print(f"Global Dice:       {global_dice:.4f}")
    print(f"Global IoU:        {global_iou:.4f}")
    print(f"Global Precision:  {global_precision:.4f}")
    print(f"Global Recall:     {global_recall:.4f}")
    print(f"Mean image Dice:   {np.mean(per_image_dice):.4f}")
    print(f"Mean image IoU:    {np.mean(per_image_iou):.4f}")
    print(f"Mean image Precision: {np.mean(per_image_precision):.4f}")
    print(f"Mean image Recall:    {np.mean(per_image_recall):.4f}")
    print("Visualizations:", OUTPUT_DIR)
    print("Metrics CSV:", csv_path)


if __name__ == "__main__":
    main()