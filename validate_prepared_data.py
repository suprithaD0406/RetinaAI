
from pathlib import Path

import numpy as np
from PIL import Image


DATASET_DIR = Path("prepared_data")
SPLITS = ("train", "validation", "test")
IMAGE_SIZE = (512, 512)


def main():
    total_pairs = 0
    errors = []

    print("PREPARED DATASET VERIFICATION")
    print("=" * 45)

    for split in SPLITS:
        image_dir = DATASET_DIR / split / "images"
        mask_dir = DATASET_DIR / split / "masks"

        if not image_dir.is_dir() or not mask_dir.is_dir():
            errors.append(f"{split}: directory missing")
            continue

        images = {
            p.stem: p
            for p in image_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        }

        masks = {
            p.stem: p
            for p in mask_dir.iterdir()
            if p.suffix.lower() in {".png", ".tif", ".tiff"}
        }

        image_ids = set(images)
        mask_ids = set(masks)
        matched_ids = image_ids & mask_ids

        missing_masks = sorted(image_ids - mask_ids)
        missing_images = sorted(mask_ids - image_ids)

        split_errors = 0
        foreground_pixels = 0
        total_pixels = 0

        for image_id in sorted(matched_ids):
            try:
                with Image.open(images[image_id]) as img:
                    image_size = img.size

                with Image.open(masks[image_id]) as mask_img:
                    mask = np.asarray(mask_img.convert("L"))

                if image_size != IMAGE_SIZE:
                    errors.append(
                        f"{split}/{image_id}: "
                        f"incorrect image size {image_size}"
                    )
                    split_errors += 1

                if mask.shape != (512, 512):
                    errors.append(
                        f"{split}/{image_id}: "
                        f"incorrect mask shape {mask.shape}"
                    )
                    split_errors += 1

                values = set(np.unique(mask).tolist())

                if not values.issubset({0, 255}):
                    errors.append(
                        f"{split}/{image_id}: "
                        f"unexpected mask values {values}"
                    )
                    split_errors += 1

                foreground_pixels += int(np.count_nonzero(mask))
                total_pixels += mask.size

            except Exception as exc:
                errors.append(
                    f"{split}/{image_id}: {exc}"
                )
                split_errors += 1

        print(f"\n{split.upper()}")
        print("-" * 35)
        print("Images:", len(images))
        print("Masks:", len(masks))
        print("Matched pairs:", len(matched_ids))
        print("Images without masks:", missing_masks)
        print("Masks without images:", missing_images)
        print("Invalid pairs:", split_errors)

        if missing_masks:
            errors.append(f"{split}: images missing masks")

        if missing_images:
            errors.append(f"{split}: masks missing images")

        percentage = (
            100 * foreground_pixels / total_pixels
            if total_pixels else 0
        )

        print(f"Foreground percentage: {percentage:.4f}%")

        total_pairs += len(matched_ids)

    print("\nFINAL RESULT")
    print("=" * 45)
    print("Total matched pairs:", total_pairs)

    if errors:
        print(f"FAILED: {len(errors)} issue(s) found.")
        for error in errors:
            print("-", error)
    elif total_pairs == 81:
        print("PASSED: All 81 image-mask pairs passed.")
    else:
        print(
            f"CHECK REQUIRED: Expected 81 pairs, found {total_pairs}."
        )


if __name__ == "__main__":
    main()
