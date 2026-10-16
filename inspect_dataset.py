
from pathlib import Path

# --------------------------------------------------
# 1. Dataset paths
# --------------------------------------------------
DATASET = Path(r"C:\Users\SUPRITHA D\Desktop\eyedata")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

MASK_CATEGORIES = [
    "1. Microaneurysms",
    "2. Haemorrhages",
    "3. Hard Exudates",
    "4. Soft Exudates",
    "5. Optic Disc",
]

# Annotation suffixes used by IDRiD
MASK_SUFFIXES = ("_MA", "_HE", "_EX", "_SE", "_OD")


# --------------------------------------------------
# 2. Helper functions
# --------------------------------------------------
def find_folder(parent, expected_name):
    """Find a folder by its name, ignoring case."""
    if not parent.exists():
        return None

    for item in parent.iterdir():
        if item.is_dir() and item.name.casefold() == expected_name.casefold():
            return item

    return None


def get_image_files(folder):
    """Collect supported image files recursively."""
    if folder is None or not folder.exists():
        return []

    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def get_image_id(path):
    """Extract an image ID from an original image filename."""
    return path.stem.upper()


def get_mask_id(path):
    """Remove the lesion annotation suffix from a mask filename."""
    stem = path.stem.upper()

    for suffix in MASK_SUFFIXES:
        if stem.endswith(suffix):
            return stem[:-len(suffix)]

    return stem


# --------------------------------------------------
# 3. Locate the dataset
# --------------------------------------------------
if not DATASET.exists():
    raise FileNotFoundError(f"Dataset folder not found: {DATASET}")

segmentation_parent = next(
    (
        p for p in DATASET.iterdir()
        if p.is_dir() and "segmentation" in p.name.lower()
    ),
    None,
)

if segmentation_parent is None:
    raise FileNotFoundError("Could not find the segmentation dataset folder.")

base = find_folder(segmentation_parent, "A. Segmentation")

if base is None:
    raise FileNotFoundError(
        f"Could not find 'A. Segmentation' inside {segmentation_parent}"
    )

original_root = find_folder(base, "1. Original Images")
groundtruth_root = find_folder(base, "2. All Segmentation Groundtruths")

if original_root is None or groundtruth_root is None:
    raise FileNotFoundError("Original Images or Groundtruths folder is missing.")


# --------------------------------------------------
# 4. Inspect training and testing sets
# --------------------------------------------------
total_images = 0
total_masks = 0
total_matches = 0

for split_name, folder_name in [
    ("TRAINING", "a. Training Set"),
    ("TESTING", "b. Testing Set"),
]:
    print(f"\n{'=' * 55}")
    print(f"{split_name} SET")
    print("=" * 55)

    image_dir = find_folder(original_root, folder_name)
    mask_split_dir = find_folder(groundtruth_root, folder_name)

    if image_dir is None:
        print(f"Original image folder not found: {folder_name}")
        continue

    if mask_split_dir is None:
        print(f"Ground-truth folder not found: {folder_name}")
        continue

    images = get_image_files(image_dir)
    image_ids = {get_image_id(p) for p in images}

    total_images += len(images)

    print("Original images:", len(images))

    for category_name in MASK_CATEGORIES:
        category_dir = find_folder(mask_split_dir, category_name)

        if category_dir is None:
            print(f"\n{category_name}: folder not found")
            continue

        masks = get_image_files(category_dir)
        mask_ids = {get_mask_id(p) for p in masks}

        matching_ids = image_ids & mask_ids
        masks_without_images = mask_ids - image_ids
        images_without_masks = image_ids - mask_ids

        total_masks += len(masks)
        total_matches += len(matching_ids)

        print(f"\n{category_name}")
        print("  Masks:", len(masks))
        print("  Matching image IDs:", len(matching_ids))
        print("  Images without masks:", len(images_without_masks))
        print("  Masks without matching images:", len(masks_without_images))

        if masks_without_images:
            print("  Unmatched mask IDs:", sorted(masks_without_images)[:5])

        if images_without_masks:
            print("  First images without masks:", sorted(images_without_masks)[:5])


# --------------------------------------------------
# 5. Summary
# --------------------------------------------------
print(f"\n{'=' * 55}")
print("INSPECTION SUMMARY")
print("=" * 55)
print("Total original images inspected:", total_images)
print("Total annotation mask files:", total_masks)
print("Total category-level matched IDs:", total_matches)
print("\nInspection complete. No dataset files were modified.")
