# RetinaAI: Retinal Hard Exudate Segmentation

A deep learning project for segmenting hard exudates in retinal fundus images using U-Net and transfer learning with EfficientNet-B0.

## Overview

RetinaAI compares three segmentation experiments: baseline U-Net, U-Net with Tversky loss, and U-Net with a pretrained EfficientNet-B0 encoder.

**Objective:** Identify hard exudate regions in retinal images using pixel-level segmentation and evaluate performance with Dice, IoU, precision, and recall.

## Models

- **Baseline U-Net:** Encoder-decoder architecture with skip connections.
- **Tversky-loss U-Net:** Experimental model designed to address class imbalance.
- **EfficientNet-B0 U-Net:** U-Net-style decoder with ImageNet-pretrained features.

## Dataset

The project uses the [IDRiD retinal fundus image dataset](https://idrid.grand-challenge.org/) and hard exudate segmentation masks.

| Split | Images |
|---|---:|
| Training | 43 |
| Validation | 11 |
| Test | 27 |
| **Total** | **81** |

Images and masks are resized to 512 × 512 pixels. Masks are binary, with background represented by 0 and exudate pixels by 255 in the prepared files.

Please follow the dataset's terms of use.

## Tech Stack

- Python 3.11
- PyTorch and Torchvision
- NumPy
- Matplotlib
- Pillow

## Evaluation Metrics

- **Dice coefficient:** Measures overlap between predicted and ground-truth regions.
- **IoU:** Measures intersection relative to union.
- **Precision:** Measures the proportion of predicted positive pixels that are correct.
- **Recall:** Measures the proportion of ground-truth positive pixels detected.

Pixel accuracy is also calculated but should be interpreted cautiously because exudates occupy a small fraction of each image.

## Experimental Results

### Best Validation Dice

| Experiment | Best Validation Dice |
|---|---:|
| Baseline U-Net | 0.5503 |
| Tversky-loss U-Net | 0.3788 |
| EfficientNet-B0 U-Net | 0.5439 |

The baseline U-Net achieved the highest recorded validation Dice.

### Baseline U-Net Test Results

Existing test evaluation at threshold 0.5:

| Metric | Score |
|---|---:|
| Global Dice | 0.5258 |
| Global IoU | 0.3567 |
| Pixel precision | 0.6037 |
| Pixel recall | 0.4657 |
| Pixel accuracy | 0.9909 |

These results apply to the current checkpoint, data split, preprocessing pipeline, and threshold. The pretrained and Tversky-loss models have not been evaluated on the test set in this comparison.

## Project Structure

```text
RetinaAI/
├── prepared_data/
├── checkpoints/
├── checkpoints_pretrained/
├── checkpoints_tversky/
├── model.py
├── model_pretrained.py
├── dataset.py
├── train.py
├── train_pretrained.py
├── train_tversky.py
├── evaluate.py
├── evaluate_pretrained.py
├── compare_validation.py
├── compare_experiments.py
├── prepare_dataset.py
├── validate_prepared_data.py
├── verify_prepared_dataset.py
├── inspect_dataset.py
├── threshold_analysis.py
├── visualize_mask.py
└── README.md
```

Evaluation outputs and comparison charts are saved in their respective output directories.

## Setup

Run these commands in Windows PowerShell from the project root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision numpy matplotlib pillow
```

Install any additional dependencies required by your scripts.

## Running the Project

Verify the prepared dataset:

```powershell
python .\verify_prepared_dataset.py
python .\validate_prepared_data.py
```

Train the baseline U-Net:

```powershell
python .\train.py
```

Evaluate the baseline model:

```powershell
python .\evaluate.py
```

Train the pretrained model:

```powershell
python .\train_pretrained.py
```

Evaluate it on validation data:

```powershell
python .\evaluate_pretrained.py
```

Compare validation results:

```powershell
python .\compare_validation.py
python .\compare_experiments.py
```

Check each script's configuration before running it, especially the checkpoint path, data split, threshold, and output directory.

## Limitations

- Only 43 training images are available.
- Validation uses just 11 images, so results have substantial uncertainty.
- Sparse exudate regions create class imbalance.
- Performance depends on preprocessing, threshold selection, and the data split.
- The project has not been clinically validated.

## Future Improvements

- Explore Attention U-Net and U-Net++.
- Improve augmentation and imbalance-aware loss functions.
- Report per-image metrics and variability.
- Establish a reproducible evaluation protocol.
- Develop an inference interface.

## Disclaimer

RetinaAI is intended for educational and research purposes only. It must not be used for clinical diagnosis or treatment decisions.
