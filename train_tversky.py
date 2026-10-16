
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
CHECKPOINT_DIR = Path("checkpoints_tversky")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = CHECKPOINT_DIR / "best_tversky_unet.pth"
HISTORY_PATH = CHECKPOINT_DIR / "training_history.csv"

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

# Tversky: larger beta penalizes false negatives more heavily.
ALPHA = 0.3  # false-positive weight
BETA = 0.7   # false-negative weight
BCE_WEIGHT = 0.3
TVERSKY_WEIGHT = 0.7

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

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ==================================================
# 3. Loss: BCE + Tversky
# ==================================================

class BCETverskyLoss(nn.Module):
    def __init__(
        self,
        alpha=0.3,
        beta=0.7,
        bce_weight=0.3,
        tversky_weight=0.7,
        smooth=1.0,
    ):
        super().__init__()

        if abs(bce_weight + tversky_weight - 1.0) > 1e-6:
            raise ValueError("Loss weights must sum to 1.")

        self.alpha = alpha
        self.beta = beta
        self.bce_weight = bce_weight
        self.tversky_weight = tversky_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        logits = logits.float()
        targets = targets.float()

        bce_loss = self.bce(logits, targets)
        probabilities = torch.sigmoid(logits)

        probabilities = probabilities.flatten(start_dim=1)
        targets = targets.flatten(start_dim=1)

        tp = (probabilities * targets).sum(dim=1)
        fp = (probabilities * (1.0 - targets)).sum(dim=1)
        fn = ((1.0 - probabilities) * targets).sum(dim=1)

        tversky = (
            tp + self.smooth
        ) / (
            tp
            + self.alpha * fp
            + self.beta * fn
            + self.smooth
        )

        tversky_loss = 1.0 - tversky.mean()

        return (
            self.bce_weight * bce_loss
            + self.tversky_weight * tversky_loss
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
    ) / (
        union + 1e-7
    )

    return dice, iou


# ==================================================
# 5. Augmentation
# ==================================================

def augment_batch(images, masks):
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
# 6. One epoch
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

            loss = criterion(logits.float(), masks.float())

            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), GRAD_CLIP_NORM
                    )

                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), GRAD_CLIP_NORM
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
# 7. Main training
# ==================================================

def main():
    set_seed(SEED)

    print("Device:", DEVICE)
    print("Experiment: BCE + Tversky")
    print(f"Alpha: {ALPHA}, Beta: {BETA}")
    print(f"BCE weight: {BCE_WEIGHT}")
    print(f"Tversky weight: {TVERSKY_WEIGHT}")
    print("Baseline checkpoint will not be modified.")

    train_dataset = RetinalSegmentationDataset(
        root_dir=str(DATA_DIR), split="train"
    )
    val_dataset = RetinalSegmentationDataset(
        root_dir=str(DATA_DIR), split="validation"
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
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    model = UNet(
        in_channels=3,
        out_channels=1,
        base_channels=32,
    ).to(DEVICE)

    criterion = BCETverskyLoss(
        alpha=ALPHA,
        beta=BETA,
        bce_weight=BCE_WEIGHT,
        tversky_weight=TVERSKY_WEIGHT,
    )

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

    history = []
    best_val_dice = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        start_time = time.time()

        train_loss, train_dice, train_iou = run_epoch(
            model, train_loader, criterion, optimizer, scaler
        )

        val_loss, val_dice, val_iou = run_epoch(
            model, val_loader, criterion
        )

        scheduler.step(val_dice)
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - start_time

        row = {
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
        history.append(row)

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"Train loss {train_loss:.4f} | "
            f"Train Dice {train_dice:.4f} | "
            f"Val loss {val_loss:.4f} | "
            f"Val Dice {val_dice:.4f} | "
            f"Val IoU {val_iou:.4f} | "
            f"LR {current_lr:.2e} | {elapsed:.1f}s"
        )

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
                    "base_channels": 32,
                    "seed": SEED,
                    "threshold": 0.5,
                    "loss": "BCE+Tversky",
                    "alpha": ALPHA,
                    "beta": BETA,
                },
                BEST_MODEL_PATH,
            )
        else:
            epochs_without_improvement += 1

        with HISTORY_PATH.open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file, fieldnames=list(history[0].keys())
            )
            writer.writeheader()
            writer.writerows(history)

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                f"\nEarly stopping at epoch {epoch}. "
                f"Best epoch: {best_epoch}."
            )
            break

    print("\n" + "=" * 55)
    print("TVERSKY EXPERIMENT FINISHED")
    print("Best epoch:", best_epoch)
    print(f"Best validation Dice: {best_val_dice:.4f}")
    print("Checkpoint:", BEST_MODEL_PATH.resolve())
    print("History:", HISTORY_PATH.resolve())
    print("Official test set was not used.")
    print("=" * 55)


if __name__ == "__main__":
    main()

