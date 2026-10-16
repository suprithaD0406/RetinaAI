
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class RetinalSegmentationDataset(Dataset):
    def __init__(self, root_dir, split):
        self.root_dir = Path(root_dir)
        self.split = split

        self.image_dir = self.root_dir / split / "images"
        self.mask_dir = self.root_dir / split / "masks"

        if not self.image_dir.is_dir():
            raise FileNotFoundError(
                f"Image directory not found: {self.image_dir.resolve()}"
            )

        if not self.mask_dir.is_dir():
            raise FileNotFoundError(
                f"Mask directory not found: {self.mask_dir.resolve()}"
            )

        image_extensions = {".png", ".jpg", ".jpeg"}

        self.image_paths = sorted(
            path
            for path in self.image_dir.iterdir()
            if path.suffix.lower() in image_extensions
        )

        mask_extensions = {".png", ".tif", ".tiff"}

        self.mask_paths = {
            path.stem: path
            for path in self.mask_dir.iterdir()
            if path.suffix.lower() in mask_extensions
        }

        missing_masks = [
            path.stem
            for path in self.image_paths
            if path.stem not in self.mask_paths
        ]

        if missing_masks:
            raise ValueError(
                f"Images without matching masks: {missing_masks}"
            )

        if not self.image_paths:
            raise ValueError(
                f"No images found in {self.image_dir.resolve()}"
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[image_path.stem]

        with Image.open(image_path) as image_file:
            image = np.asarray(
                image_file.convert("RGB"),
                dtype=np.float32
            ) / 255.0

        with Image.open(mask_path) as mask_file:
            mask = np.asarray(
                mask_file.convert("L"),
                dtype=np.float32
            )

        # Convert the mask from 0/255 to 0/1.
        mask = (mask > 127).astype(np.float32)

        # HWC -> CHW for PyTorch.
        image = torch.from_numpy(
            image.transpose(2, 0, 1).copy()
        )

        # Add the single mask channel.
        mask = torch.from_numpy(
            mask[np.newaxis, :, :].copy()
        )

        return image, mask


def main():
    root_dir = Path("prepared_data")

    print("DEVICE:", torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ))

    for split in ("train", "validation", "test"):
        dataset = RetinalSegmentationDataset(
            root_dir=root_dir,
            split=split
        )

        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=(split == "train"),
            num_workers=0
        )

        images, masks = next(iter(loader))

        print(f"\n{split.upper()}")
        print("Dataset size:", len(dataset))
        print("Image batch shape:", tuple(images.shape))
        print("Mask batch shape:", tuple(masks.shape))
        print("Image value range:", (
            images.min().item(),
            images.max().item()
        ))
        print("Mask values:", torch.unique(masks).tolist())

    print("\nDataset loading test passed.")


if __name__ == "__main__":
    main()
