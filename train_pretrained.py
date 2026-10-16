import os
import random
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import RetinalSegmentationDataset
from model_pretrained import PretrainedUNet

# Configuration
SEED = 42
DATA_DIR = "prepared_data"
OUTPUT_DIR = "checkpoints_pretrained"
BATCH_SIZE = 1
EPOCHS = 60
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
PATIENCE = 12
NUM_WORKERS = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ImageNet normalization for the pretrained encoder
IMAGENET_MEAN = torch.tensor(
    [0.485, 0.456, 0.406]
).view(1, 3, 1, 1)

IMAGENET_STD = torch.tensor(
    [0.229, 0.224, 0.225]
).view(1, 3, 1, 1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_images(images):
    mean = IMAGENET_MEAN.to(images.device)
    std = IMAGENET_STD.to(images.device)
    return (images - mean) / std


def augment(images, masks):
    # Apply the same geometric transformation to image and mask.
    if random.random() < 0.5:
        images = torch.flip(images, dims=[3])
        masks = torch.flip(masks, dims=[3])

    if random.random() < 0.5:
        images = torch.flip(images, dims=[2])
        masks = torch.flip(masks, dims=[2])

    rotations = random.randint(0, 3)
    if rotations:
        images = torch.rot90(images, rotations, dims=[2, 3])
        masks = torch.rot90(masks, rotations, dims=[2, 3])

    return images, masks


def dice_loss(logits, masks):
    probabilities = torch.sigmoid(logits)

    intersection = (probabilities * masks).sum(dim=(1, 2, 3))
    denominator = (
        probabilities.sum(dim=(1, 2, 3))
        + masks.sum(dim=(1, 2, 3))
    )

    dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    return 1.0 - dice.mean()


def segmentation_loss(logits, masks):
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, masks
    )
    dice = dice_loss(logits, masks)
    return 0.5 * bce + 0.5 * dice


def calculate_metrics(logits, masks, threshold=0.5):
    predictions = torch.sigmoid(logits) >= threshold
    targets = masks >= 0.5

    predictions = predictions.float()
    targets = targets.float()

    intersection = (predictions * targets).sum(dim=(1, 2, 3))
    pred_sum = predictions.sum(dim=(1, 2, 3))
    target_sum = targets.sum(dim=(1, 2, 3))

    dice = (2 * intersection + 1e-7) / (
        pred_sum + target_sum + 1e-7
    )
    iou = (intersection + 1e-7) / (
        pred_sum + target_sum - intersection + 1e-7
    )

    return dice.mean().item(), iou.mean().item()


def run_epoch(model, loader, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_images = 0

    for images, masks in loader:
        images = images.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        if training:
            images, masks = augment(images, masks)

        images = normalize_images(images)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=(DEVICE.type == "cuda"),
            ):
                logits = model(images)
                loss = segmentation_loss(logits, masks)

            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                scaler.step(optimizer)
                scaler.update()

        batch_size = images.size(0)
        dice, iou = calculate_metrics(logits.detach(), masks)

        total_loss += loss.item() * batch_size
        total_dice += dice * batch_size
        total_iou += iou * batch_size
        total_images += batch_size

    return (
        total_loss / total_images,
        total_dice / total_images,
        total_iou / total_images,
    )


def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(
            f"Dataset folder not found: {DATA_DIR}"
        )

    train_dataset = RetinalSegmentationDataset(DATA_DIR, "train")
    val_dataset = RetinalSegmentationDataset(DATA_DIR, "validation")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    model = PretrainedUNet(pretrained=True).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
    )

    scaler = torch.amp.GradScaler(
        "cuda", enabled=(DEVICE.type == "cuda")
    )

    history_path = os.path.join(OUTPUT_DIR, "training_history.csv")
    best_path = os.path.join(OUTPUT_DIR, "best_pretrained_unet.pth")

    best_val_dice = -1.0
    epochs_without_improvement = 0

    print("Device:", DEVICE)
    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_dataset))
    print("Batch size:", BATCH_SIZE)
    print("Epochs:", EPOCHS)
    print("Output folder:", OUTPUT_DIR)

    with open(history_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "epoch", "train_loss", "train_dice", "train_iou",
            "val_loss", "val_dice", "val_iou", "learning_rate"
        ])

        for epoch in range(1, EPOCHS + 1):
            train_loss, train_dice, train_iou = run_epoch(
                model, train_loader, optimizer, scaler
            )

            val_loss, val_dice, val_iou = run_epoch(
                model, val_loader
            )

            scheduler.step(val_dice)
            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch:02d}/{EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Dice: {train_dice:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Dice: {val_dice:.4f} | "
                f"Val IoU: {val_iou:.4f} | "
                f"LR: {current_lr:.2e}"
            )

            writer.writerow([
                epoch, train_loss, train_dice, train_iou,
                val_loss, val_dice, val_iou, current_lr
            ])
            file.flush()

            if val_dice > best_val_dice:
                best_val_dice = val_dice
                epochs_without_improvement = 0

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_dice": best_val_dice,
                    },
                    best_path,
                )

                print(f"  Saved best checkpoint: {best_path}")
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= PATIENCE:
                print("Early stopping triggered.")
                break

    print("\nTraining complete.")
    print("Best validation Dice:", f"{best_val_dice:.4f}")
    print("Best checkpoint:", best_path)
    print("Training history:", history_path)


if __name__ == "__main__":
    main()