
from pathlib import Path
import csv
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import RetinalSegmentationDataset
from model import UNet


# ==================================================
# 1. Configuration
# ==================================================

SEED = 42

DATA_DIR = Path("prepared_data")
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = CHECKPOINT_DIR / "best_unet.pth"
HISTORY_PATH = CHECKPOINT_DIR / "training_history.csv"

IMAGE_SIZE = 512
BATCH_SIZE = 1
MAX_EPOCHS = 60
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 12
LR_PATIENCE = 4
MIN_LR = 1e-6
NUM_WORKERS = 0
USE_AMP = True
GRAD_CLIP_NORM = 1.0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==================================================
# 2. Reproducibility
# ==================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Reproducible behavior where supported.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ==================================================
# 3. Loss: BCE + soft Dice
# ==================================================

class BCEDiceLoss(nn.Module):
    def __init__(self, dice_weight=0.5, smooth=1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        # Compute loss reductions in float32 for stability.
        logits = logits.float()
        targets = targets.float()

        bce_loss = self.bce(logits, targets)
        probabilities = torch.sigmoid(logits)

        probabilities = probabilities.flatten(start_dim=1)
        targets = targets.flatten(start_dim=1)

        intersection = (probabilities * targets).sum(dim=1)

        dice = (
            2.0 * intersection + self.smooth
        ) / (
            probabilities.sum(dim=1)
            + targets.sum(dim=1)
            + self.smooth
        )

        dice_loss = 1.0 - dice.mean()

        return (
            (1.0 - self.dice_weight) * bce_loss
            + self.dice_weight * dice_loss
        )


# ==================================================
# 4. Metrics
# ==================================================

@torch.no_grad()
def calculate_metrics(logits, targets, threshold=0.5):
    probabilities = torch.sigmoid(logits.float())
    predictions = probabilities >= threshold
    targets = targets >= 0.5

    predictions = predictions.flatten(start_dim=1)
    targets = targets.flatten(start_dim=1)

    intersection = (predictions & targets).sum(dim=1).float()
    pred_count = predictions.sum(dim=1).float()
    target_count = targets.sum(dim=1).float()

    dice = (
        2 * intersection + 1e-7
    ) / (
        pred_count + target_count + 1e-7
    )

    union = (predictions | targets).sum(dim=1).float()

    iou = (
        intersection + 1e-7
    ) / (union + 1e-7)

    return dice, iou


# ==================================================
# 5. Safe geometric augmentation
# ==================================================

def augment_batch(images, masks):
    # Apply the same transformation to each image and its mask.
    # Flips and right-angle rotations preserve binary labels.
    for i in range(images.size(0)):
        if random.random() < 0.5:
            images[i] = torch.flip(images[i], dims=[2])
            masks[i] = torch.flip(masks[i], dims=[2])

        if random.random() < 0.5:
            images[i] = torch.flip(images[i], dims=[1])
            masks[i] = torch.flip(masks[i], dims=[1])

        rotations = random.randint(0, 3)
        if rotations:
            images[i] = torch.rot90(
                images[i], rotations, dims=[1, 2]
            )
            masks[i] = torch.rot90(
                masks[i], rotations, dims=[1, 2]
            )

    return images, masks


# ==================================================
# 6. One training or validation epoch
# ==================================================

def run_epoch(model, loader, criterion, optimizer=None, scaler=None):
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_samples = 0

    for images, masks in loader:
        images = images.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        if training:
            images, masks = augment_batch(images, masks)
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=(USE_AMP and DEVICE.type == "cuda"),
            ):
                logits = model(images)

            # Calculate loss outside autocast for numerical stability.
            loss = criterion(logits.float(), masks.float())

            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        GRAD_CLIP_NORM
                    )

                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        GRAD_CLIP_NORM
                    )

                    optimizer.step()

        dice, iou = calculate_metrics(logits.detach(), masks)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_dice += dice.mean().item() * batch_size
        total_iou += iou.mean().item() * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError("DataLoader returned no samples.")

    return (
        total_loss / total_samples,
        total_dice / total_samples,
        total_iou / total_samples,
    )


# ==================================================
# 7. Preflight: test forward, loss, backward and update
# ==================================================

def preflight_check(model, loader, criterion, optimizer, scaler):
    print("\nRunning preflight checks before training...")

    model.train()
    images, masks = next(iter(loader))

    images = images.to(DEVICE)
    masks = masks.to(DEVICE)

    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(
        device_type=DEVICE.type,
        dtype=torch.float16,
        enabled=(USE_AMP and DEVICE.type == "cuda"),
    ):
        logits = model(images)

    if logits.shape != masks.shape:
        raise RuntimeError(
            f"Shape mismatch: logits {logits.shape}, masks {masks.shape}"
        )

    loss = criterion(logits.float(), masks.float())

    if not torch.isfinite(loss):
        raise RuntimeError("Preflight failed: loss is not finite.")

    if scaler is not None and scaler.is_enabled():
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
    else:
        loss.backward()

    gradients_found = any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    if not gradients_found:
        raise RuntimeError(
            "Preflight failed: no finite gradients found."
        )

    torch.nn.utils.clip_grad_norm_(
        model.parameters(), GRAD_CLIP_NORM
    )

    if scaler is not None and scaler.is_enabled():
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    optimizer.zero_grad(set_to_none=True)

    print("Preflight passed.")
    print("Input shape:", tuple(images.shape))
    print("Output shape:", tuple(logits.shape))
    print("Initial loss:", f"{loss.item():.4f}")


# ==================================================
# 8. Save CSV history
# ==================================================

def save_history(history):
    with HISTORY_PATH.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_dice",
                "train_iou",
                "val_loss",
                "val_dice",
                "val_iou",
                "learning_rate",
                "epoch_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(history)


# ==================================================
# 9. Main
# ==================================================

def main():
    set_seed(SEED)

    print("=" * 55)
    print("RETINAAI - U-NET TRAINING")
    print("=" * 55)
    print("Device:", DEVICE)

    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print(
            "GPU memory:",
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    train_dataset = RetinalSegmentationDataset(
        DATA_DIR, "train"
    )
    val_dataset = RetinalSegmentationDataset(
        DATA_DIR, "validation"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_dataset))
    print("Batch size:", BATCH_SIZE)
    print("Maximum epochs:", MAX_EPOCHS)
    print("Mixed precision:", USE_AMP and DEVICE.type == "cuda")

    model = UNet(
        in_channels=3,
        out_channels=1,
        base_channels=32,
    ).to(DEVICE)

    criterion = BCEDiceLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=LR_PATIENCE,
        min_lr=MIN_LR,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(USE_AMP and DEVICE.type == "cuda"),
    )

    # Fail early if forward/backward/update does not work.
    preflight_check(
        model, train_loader, criterion, optimizer, scaler
    )

    best_val_dice = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        start = time.time()

        train_loss, train_dice, train_iou = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer=optimizer,
            scaler=scaler,
        )

        val_loss, val_dice, val_iou = run_epoch(
            model,
            val_loader,
            criterion,
        )

        scheduler.step(val_dice)
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - start

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_dice": train_dice,
            "train_iou": train_iou,
            "val_loss": val_loss,
            "val_dice": val_dice,
            "val_iou": val_iou,
            "learning_rate": current_lr,
            "epoch_seconds": elapsed,
        }

        history.append(record)
        save_history(history)

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"Train loss {train_loss:.4f} | "
            f"Train Dice {train_dice:.4f} | "
            f"Train IoU {train_iou:.4f} | "
            f"Val loss {val_loss:.4f} | "
            f"Val Dice {val_dice:.4f} | "
            f"Val IoU {val_iou:.4f} | "
            f"LR {current_lr:.2e} | "
            f"{elapsed:.1f}s"
        )

        # Save only when validation Dice improves.
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "best_val_dice": val_dice,
                    "val_loss": val_loss,
                    "val_iou": val_iou,
                    "seed": SEED,
                    "image_size": IMAGE_SIZE,
                    "base_channels": 32,
                },
                BEST_MODEL_PATH,
            )

            print("  -> Saved best checkpoint.")
        else:
            epochs_without_improvement += 1

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print("\nEarly stopping: validation Dice did not improve.")
            break

    print("\n" + "=" * 55)
    print("TRAINING FINISHED")
    print("=" * 55)
    print("Best epoch:", best_epoch)
    print("Best validation Dice:", f"{best_val_dice:.4f}")
    print("Best checkpoint:", BEST_MODEL_PATH.resolve())
    print("Training history:", HISTORY_PATH.resolve())
    print("The official test set was not used during training.")


if __name__ == "__main__":
    try:
        main()
    except torch.cuda.OutOfMemoryError:
        print("\nCUDA OUT OF MEMORY.")
        print(
            "Training stopped. No automatic restart was attempted. "
            "The batch size is already 1; check GPU memory and other "
            "GPU applications before changing the model."
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise
