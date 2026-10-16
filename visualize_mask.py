
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

DATASET = Path(r"C:\Users\SUPRITHA D\Desktop\eyedata")

segmentation = next(
    p for p in DATASET.iterdir()
    if "segmentation" in p.name.lower()
)

base = segmentation / "A. Segmentation"

image_path = (
    base / "1. Original Images" / "a. Training Set" / "IDRiD_01.jpg"
)

mask_path = (
    base / "2. All Segmentation Groundtruths"
    / "a. Training Set" / "3. Hard Exudates" / "IDRiD_01_EX.tif"
)

if not image_path.exists():
    raise FileNotFoundError(f"Image not found: {image_path}")

if not mask_path.exists():
    raise FileNotFoundError(f"Mask not found: {mask_path}")

image = np.asarray(Image.open(image_path).convert("RGB"))
mask = np.asarray(Image.open(mask_path))

# Convert the binary mask to a Boolean array.
mask = mask > 0
print("Image dimensions:", image.shape[:2])
print("Mask dimensions:", mask.shape)
print("Dimensions match:", image.shape[:2] == mask.shape)
print("Mask contains lesion pixels:", np.any(mask))
print("Lesion pixel percentage:", round(mask.mean() * 100, 4), "%")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image)
axes[0].set_title("Original Retinal Image")

axes[1].imshow(mask, cmap="gray")
axes[1].set_title("Hard Exudates Ground Truth")

axes[2].imshow(image)
overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
overlay[mask] = [1, 0, 0, 0.7]
axes[2].imshow(overlay)
axes[2].set_title("Mask Overlay")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()
