
from pathlib import Path
import random

import numpy as np
from PIL import Image

# ==================================================
# 1. CONFIGURATION
# ==================================================
DATASET = Path(r"C:\Users\SUPRITHA D\Desktop\eyedata")
OUTPUT = Path("prepared_data")

IMAGE_SIZE = (512, 512)  # width, height
VALIDATION_RATIO = 0.20
SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


# ==================================================
# 2. FOLDER HELPERS
# ==================================================
def find_folder(parent, name):
    if not parent.exists():
        raise FileNotFoundError(f"Folder not found: {parent}")

    for item in parent.iterdir():
        if item.is_dir() and item.name.casefold() == name.casefold():
            return item

    raise FileNotFoundError(f"Folder '{name}' not found inside {parent}")


def get_files(folder):
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


# ==================================================
# 3. LOCATE DATASET
# ==================================================
if not DATASET.exists():
    raise FileNotFoundError(f"Dataset not found: {DATASET}")

segmentation_parent = next(
    (
        p for p in DATASET.iterdir()
        if p.is_dir() and "segmentation" in p.name.lower()
    ),
    None,
)

if segmentation_parent is None:
    raise FileNotFoundError("Could not locate the segmentation dataset.")

base = find_folder(segmentation_parent, "A. Segmentation")
original_root = find_folder(base, "1. Original Images")
groundtruth_root = find_folder(base, "2. All Segmentation Groundtruths")


# ==================================================
# 4. MATCH IMAGES AND HARD EXUDATE MASKS
# ==================================================
def get_pairs(split_name, image_folder_name, mask_folder_name):
    image_dir = find_folder(original_root, image_folder_name)
    mask_split = find_folder(groundtruth_root, mask_folder_name)
    mask_dir = find_folder(mask_split, "3. Hard Exudates")

    images = {p.stem.upper(): p for p in get_files(image_dir)}

    masks = {
        p.stem.upper().removesuffix("_EX"): p
        for p in get_files(mask_dir)
    }

    common_ids = sorted(images.keys() & masks.keys())

    print(f"\n{split_name}:")
    print("  Original images:", len(images))
    print("  Hard Exudate masks:", len(masks))
    print("  Matched pairs:", len(common_ids))
    print("  Images without masks:", len(images.keys() - masks.keys()))
    print("  Masks without images:", len(masks.keys() - images.keys()))

    return [(images[i], masks[i], i) for i in common_ids]


training_pairs = get_pairs(
    "Official training set",
    "a. Training Set",
    "a. Training Set",
)

testing_pairs = get_pairs(
    "Official testing set",
    "b. Testing Set",
    "b. Testing Set",
)

if len(training_pairs) < 2 or not testing_pairs:
    raise RuntimeError("Insufficient matched image-mask pairs.")


# ==================================================
# 5. TRAIN / VALIDATION SPLIT
# ==================================================
rng = random.Random(SEED)
training_pairs = training_pairs.copy()
rng.shuffle(training_pairs)

validation_count = max(
    1, round(len(training_pairs) * VALIDATION_RATIO)
)

validation_pairs = training_pairs[:validation_count]
train_pairs = training_pairs[validation_count:]

print("\nPrepared split sizes:")
print("  Training:", len(train_pairs))
print("  Validation:", len(validation_pairs))
print("  Testing:", len(testing_pairs))


# ==================================================
# 6. CONVERT MASK TO BINARY SAFELY
# ==================================================
def convert_mask_to_binary(mask_image):
    """
    Convert an IDRiD ground-truth mask into a binary mask.

    IDRiD masks may be grayscale, palette-based, or RGB.
    For palette masks, convert through RGB so that palette
    indices are not mistakenly interpreted as lesion labels.

    The expected IDRiD annotation uses black background and
    colored lesion annotations. If the encoding differs,
    inspect it before using the prepared masks for training.
    """
    rgb = np.asarray(mask_image.convert("RGB"))

    # A pixel is annotated if any RGB channel is nonzero.
    # This assumes the background is black.
    foreground = np.any(rgb > 0, axis=-1)

    return (foreground.astype(np.uint8) * 255)


# ==================================================
# 7. PREPROCESS AND SAVE
# ==================================================
def prepare_split(pairs, split_name):
    image_output = OUTPUT / split_name / "images"
    mask_output = OUTPUT / split_name / "masks"

    image_output.mkdir(parents=True, exist_ok=True)
    mask_output.mkdir(parents=True, exist_ok=True)

    for image_path, mask_path, image_id in pairs:

        # Load and resize original image.
        with Image.open(image_path) as img:
            image = img.convert("RGB")
            image = image.resize(
                IMAGE_SIZE,
                Image.Resampling.LANCZOS,
            )
            image.save(image_output / f"{image_id}.png")

        # Load and resize mask.
        with Image.open(mask_path) as mask_img:
            mask = convert_mask_to_binary(mask_img)

            binary_mask = Image.fromarray(mask)
            binary_mask = binary_mask.resize(
                IMAGE_SIZE,
                Image.Resampling.NEAREST,
            )
            binary_mask.save(mask_output / f"{image_id}.png")

    print(f"Saved {len(pairs)} pairs to {OUTPUT / split_name}")


prepare_split(train_pairs, "train")
prepare_split(validation_pairs, "validation")
prepare_split(testing_pairs, "test")


# ==================================================
# 8. VERIFY OUTPUT
# ==================================================
print("\n" + "=" * 50)
print("FINAL DATASET VERIFICATION")
print("=" * 50)

for split_name in ("train", "validation", "test"):
    image_dir = OUTPUT / split_name / "images"
    mask_dir = OUTPUT / split_name / "masks"

    images = sorted(image_dir.glob("*.png"))
    masks = sorted(mask_dir.glob("*.png"))

    image_ids = {p.stem for p in images}
    mask_ids = {p.stem for p in masks}

    print(f"\n{split_name.upper()}")
    print("  Images:", len(images))
    print("  Masks:", len(masks))
    print("  Filenames match:", image_ids == mask_ids)

    if image_ids != mask_ids:
        print("  Missing masks:", sorted(image_ids - mask_ids))
        print("  Missing images:", sorted(mask_ids - image_ids))

    if images and masks:
        with Image.open(images[0]) as img, Image.open(masks[0]) as msk:
            mask_array = np.asarray(msk)

            print("  Image dimensions:", img.size)
            print("  Mask dimensions:", msk.size)
            print("  Mask values:", np.unique(mask_array).tolist())
            print(
                "  Foreground pixel percentage:",
                round(np.mean(mask_array > 0) * 100, 4),
                "%",
            )

            if img.size != msk.size:
                raise ValueError(f"Dimension mismatch in {split_name}.")

            if not set(np.unique(mask_array)).issubset({0, 255}):
                raise ValueError(f"Unexpected mask values in {split_name}.")

print("\nPreprocessing completed.")
print("Original dataset was not modified.")
