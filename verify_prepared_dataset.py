from pathlib import Path
import numpy as np
from PIL import Image

DATASET_DIR = Path("prepared_data")
SPLITS = ("train", "validation", "test")

def main():
    total_pairs = 0
    errors = []

    print("PREPARED DATASET VERIFICATION")
    print("=" * 40)

    for split in SPLITS:
        image_dir = DATASET_DIR / split / "images"
        mask_dir = DATASET_DIR / split / "masks"

        if not image_dir.is_dir() or not mask_dir.is_dir():
            errors.append(f"{split}: directory missing")
            print(f"{split}: directory missing")
            continue

        images = {
            p.stem: p for p in image_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        }
        masks = {
            p.stem: p for p in mask_dir.iterdir()
            if p.suffix.lower() in {".png", ".tif", ".tiff"}
        }

        matched = set(images) & set(masks)
        missing_masks = sorted(set(images) - set(masks))
        missing_images = sorted(set(masks) - set(images))
        invalid = 0
        foreground = 0
        pixel_count = 0

        for name in sorted(matched):
            try:
                with Image.open(images[name]) as img:
                    image_size = img.size
                with Image.open(masks[name]) as mask_img:
                    mask = np.asarray(mask_img.convert("L"))

                if image_size != (512, 512):
                    errors.append(f"{split}/{name}: wrong image size")
                    invalid += 1

                if mask.shape != (512, 512):
                    errors.append(f"{split}/{name}: wrong mask size")
                    invalid += 1

                if not set(np.unique(mask).tolist()).issubset({0, 255}):
                    errors.append(f"{split}/{name}: unexpected mask values")
                    invalid += 1

                foreground += int(np.count_nonzero(mask))
                pixel_count += mask.size

            except Exception as exc:
                errors.append(f"{split}/{name}: {exc}")
                invalid += 1

        total_pairs += len(matched)

        print(f"\n{split.upper()}")
        print("Images:", len(images))
        print("Masks:", len(masks))
        print("Matched pairs:", len(matched))
        print("Images without masks:", missing_masks)
        print("Masks without images:", missing_images)
        print("Invalid pairs:", invalid)
        print(
            "Foreground percentage:",
            f"{100 * foreground / pixel_count:.4f}%"
            if pixel_count else "N/A"
        )

        if missing_masks or missing_images:
            errors.append(f"{split}: unmatched files")

    print("\nTOTAL MATCHED PAIRS:", total_pairs)

    if not errors and total_pairs == 81:
        print("PASSED: All 81 image-mask pairs verified.")
    else:
        print("CHECK REQUIRED. Issues:", len(errors))
        for error in errors:
            print("-", error)

if __name__ == "__main__":
    main()
