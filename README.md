# Adversarial Patch Training for YOLO Models

This repository contains Python scripts to train and evaluate adversarial patches designed to deceive YOLO (You Only Look Once) object detection models. The primary goal is to generate patches that, when placed over objects in an image, cause the model to fail in detecting them (a "hiding" attack).

The project includes two main modes:
1.  **Normal Mode**: Focuses purely on the adversarial objective of hiding the object from the detector.
2.  **Covert Mode**: A two-stage optimization process that balances the adversarial objective with a visual camouflage pattern, making the patch less conspicuous.

## Features

-   **Multi-Model Training**: Train a single patch against an ensemble of YOLO models to improve robustness.
-   **Covert Camouflage Generation**: Utilizes Perlin noise to generate procedural camouflage patterns adapted to the background of training images, creating visually deceptive and context-aware patches.
-   **Advanced Training Techniques**:
    -   Automatic batch size tuning to maximize VRAM usage.
    -   Multiple learning rate schedulers (`CosineAnnealingWarmRestarts`, `CosineAnnealingLR`, `ReduceLROnPlateau`).
    -   Mixed-precision training for improved performance.
    -   Optional `torch.compile()` for model optimization.
-   **Rich CLI Interface**: Employs the `rich` library for a clean, informative, and interactive command-line experience during training and evaluation.
-   **Comprehensive Evaluation**: The `evaluate_patch.py` script provides a detailed breakdown of the patch's effectiveness, categorizing attack outcomes as Hidden, Misclassified, or Disrupted.

## File Structure

```
.
├── train_patch.py         # Main script for training the adversarial patch
├── evaluate_patch.py      # Script to evaluate the performance of a trained patch
├── requirements.txt       # All necessary Python packages
├── config.json            # Configuration file for training parameters
└── yolov11n.pt            # Example YOLO model weights (user-provided)
```

## Setup

### 1. Clone the Repository

```bash
git clone <your-repo-link>
cd <your-repo-directory>
```

### 2. Create a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies

Install all the required packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4. Download Dataset

This script is configured to use the **VisDrone2019-DET-train** dataset. You can find the dataset and download instructions on the official [VisDrone GitHub page](https://github.com/VisDrone/VisDrone-Dataset). Download it and place it in the root directory of this project. The expected directory structure is:

```
.
├── VisDrone2019-DET-train/
│   ├── images/
│   │   ├── 0000001.jpg
│   │   └── ...
│   └── annotations_v11/
│       ├── 0000001.txt
│       └── ...
└── ...
```

### 5. Download YOLO Models

Download the YOLO model weights you wish to target and place them in the root directory. You can specify which models to use in the `config.json` file.

## Usage

### 1. Configure the Training

Modify the `config.json` file to set up your training run.

-   `models_to_target`: A list of paths to the YOLO models (e.g., `["yolov11n.pt", "yolov8s.pt"]`).
-   `dataset_path`: Path to the training dataset directory.
-   `target_classes`: A list of class names to target (e.g., `["car", "person"]`). Leave empty to target all classes.
-   `patch_size`: The resolution of the patch to be trained (e.g., `150` for a 150x150 patch).
-   `hyperparameters`: Set learning rate, batch size, epochs, and scheduler settings.
-   `loss_weights`: Adjust the weights for different components of the loss function.

> **Note on Loss Weights**: The default `loss_weights` in `config.json` are tuned for the `--covert` mode. For normal mode training, you may need to adjust these values for optimal performance. Specifically, setting `pattern_weight` to `0.0` and potentially increasing `adv_weight` is a good starting point.

### 2. Train the Patch

Run the `train_patch.py` script to begin training.

**Basic Training:**
```bash
python train_patch.py
```

**Covert Mode Training:**
To enable the camouflage generation, use the `--covert` flag.
```bash
python train_patch.py --covert
```

**Available Arguments:**
-   `--batch_size`: Manually set the batch size and skip auto-tuning.
-   `--resume`: Path to a checkpoint to resume training.
-   `--starter_image`: Path to an image to use as the starting point for the patch.
-   `--gpu_ids`: Specify which GPU IDs to use (e.g., `--gpu_ids 0 1`).
-   `--patches`: Number of patches to generate in sequence.
-   `--scheduler`: Choose the learning rate scheduler (`plateau`, `cosine`, `cosine_warm`).
-   `--no-tv-loss`: Disable the total variation loss.
-   `--no-color-loss`: Disable the color diversity loss.
-   `--no-compile`: Disable `torch.compile()`.
-   `--patch_coverage`: Set the desired patch coverage of the target object's area (default: 35%).

Training progress, logs, and the final patch (`best_patch.png`) will be saved in a new directory inside `runs/`.

### 3. Evaluate the Patch

After training, use the `evaluate_patch.py` script to test its effectiveness on a validation set.

**Evaluation Command:**
```bash
python evaluate_patch.py --model yolov11n.pt --patch runs/<your-run-directory>/best_patch.png
```

**Evaluation Arguments:**
-   `--model`: Path to the YOLO model to evaluate against.
-   `--patch`: Path to the trained adversarial patch.
-   `--output`: Directory to save comparison images.
-   `--target_classes`: Specify which classes to attack during evaluation.
-   `--coverage`: Set the patch coverage percentage for evaluation.
-   `--conf`: Confidence threshold for detection.
-   `--iou`: IoU threshold for NMS.

The script will output a summary table detailing the success rate and save side-by-side comparison images in the specified output directory.
