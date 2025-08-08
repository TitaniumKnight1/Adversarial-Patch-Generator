# Adversarial Patch Generator

This repository provides a comprehensive suite of Python scripts for training, evaluating, and optimizing adversarial patches designed to deceive YOLO (You Only Look Once) object detection models. The primary goal is to generate patches that, when placed over objects in an image, cause a model to fail in its detection task (e.g., a "hiding" attack).

The toolkit is built for flexibility and performance, offering multiple training backends, various attack strategies, and automated hyperparameter tuning.

## Key Features

* **Multiple Training Backends**:
    * `train_patch.py`: A feature-rich script for training on a single node (supports single or multiple GPUs).
    * `train_patch_MG.py`: A high-performance distributed training script using `torchrun` and DDP for maximum speed on multi-GPU servers.
* **Advanced Training Strategies**:
    * **Normal Mode**: Focuses purely on the adversarial objective.
    * **Covert Style Mode**: Balances the adversarial goal with a VGG-based style loss to mimic the texture of the surrounding environment.
    * **Covert Procedural Mode**: Uses Perlin noise to generate procedural camouflage patterns for the patch.
* **Automated Hyperparameter Tuning**:
    * `automate_tuning.py`: A script to systematically test a grid of parameters, run shortened training/evaluation cycles, and find the optimal settings for your patch.
* **Comprehensive Evaluation**:
    * `evaluate_patch.py`: A detailed script to measure patch effectiveness, categorizing outcomes as **Hidden**, **Misclassified**, or **Disrupted**.
* **Modern Tooling**:
    * **Rich CLI**: Employs the `rich` library for a clean, informative, and interactive command-line experience.
    * **Automatic Batch Size Tuning**: Maximizes VRAM usage and throughput automatically.
    * **Advanced Augmentations**: Uses image transforms to simulate real-world conditions.

## File Structure

```
.
├── train_patch.py              # Main script for single-node training
├── train_patch_MG.py           # High-performance script for multi-GPU (DDP) training
├── evaluate_patch.py           # Script to evaluate a trained patch's performance
├── automate_tuning.py          # Script for automated hyperparameter tuning
├── config.json                 # Central configuration file for all training parameters
├── requirements.txt            # All necessary Python packages
└── VisDrone2019-DET-train/     # Example dataset directory (user-provided)
└── runs/                       # Output directory for logs, checkpoints, and patches
```

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/Adversarial-Patch-Generator.git
cd Adversarial-Patch-Generator
```

### 2. Create a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies

Install all the required packages from the provided `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4. Download Dataset

This project is configured to use the **VisDrone2019** dataset. Download it and place it in the root directory of this project. The expected directory structure is:

```
.
├── VisDrone2019-DET-train/
│   ├── images/
│   └── annotations_v11/
├── VisDrone2019-DET-val/
│   ├── images/
│   └── annotations_v11/
└── ...
```

### 5. Download YOLO Models

Download the YOLO model weights you wish to target (e.g., `yolov11n.pt`) and place them in the root directory. You will specify which models to use in `config.json`.

## Core Configuration (`config.json`)

This file is the central hub for configuring all training runs. Below is a detailed breakdown of the parameters within each training mode.

### Hyperparameters
These control the overall training process.
* `base_learning_rate`: Controls how much the patch is updated in each step. A higher value means faster changes but can become unstable.
    * **Recommended Range**: `0.001` to `0.05`. A good starting point is `0.01`.
* `base_batch_size`: The number of images processed at once. The script autotunes this, so this value is mainly a fallback.
* `max_epochs`: The total number of times the training process will iterate over the entire dataset.
    * **Recommended Range**: `80` for quick tests, `200-500` for full training runs.
* `plateau_patience`: For the `plateau` scheduler, this is the number of epochs to wait for the loss to improve before reducing the learning rate.
    * **Recommended Range**: `10` to `25`.
* `early_stopping_patience`: The number of epochs to wait for the loss to improve before stopping the training entirely. This prevents wasting time on a run that is not converging.
    * **Recommended Range**: `25` to `50`.
* `cosine_restart_epochs`: For the `cosine_warm` scheduler, this is the number of epochs in the first cycle before the learning rate is "restarted" to its initial value.
    * **Recommended Range**: `50` to `100`.

### Loss Weights
These values balance the different objectives of the training. The final loss is a weighted sum of these components.
* `adv_weight`: The weight of the main adversarial loss. This is the most critical component, driving the patch to hide objects.
    * **Recommended Range**: `100.0` to `1000.0`. Higher values make the patch more aggressive.
* `style_weight`: (*covert_style mode only*) Controls how strongly the patch tries to mimic the background texture. Very high values are often needed to compete with the adversarial loss.
    * **Recommended Range**: `1e4` to `1e6`.
* `nps_weight`: Weight for the Non-Printability Score, which encourages the patch to use colors that are easy to reproduce on a physical printer.
    * **Recommended Range**: `0.01` to `1.0`. A small value is usually sufficient.
* `pattern_weight`: (*covert_procedural mode only*) Controls how closely the patch should match the generated Perlin noise camouflage pattern.
    * **Recommended Range**: `100.0` to `500.0`.
* `tv_weight`: Weight for the Total Variation loss, which encourages smoothness in the patch and reduces pixelated noise.
    * **Recommended Range**: `1e-5` to `1e-3`.

### Noise Parameters
These are used for the `normal` and `covert_procedural` modes to generate Perlin noise patterns. Use the `automate_tuning.py` script to find the best combination.
* `persistence`: Controls the roughness. Higher values increase the amplitude of finer details.
    * **Recommended Range**: `0.4` to `0.7`.
* `lacunarity`: Controls the level of detail. Higher values increase the frequency of the noise.
    * **Recommended Range**: `1.8` to `2.2`.
* `scale`: Controls the overall "zoom" level of the noise pattern.
    * **Recommended Range**: `25.0` to `100.0`.
* `octaves`: The number of noise layers that are combined. More octaves add more detail but are computationally more expensive.
    * **Recommended Range**: `4` to `12`.

## Usage Guide

### 1. Standard Training (`train_patch.py`)

This is the primary script for most use cases on a single machine (with one or more GPUs).

**Command Examples:**

```bash
# Train a patch in 'normal' mode
python train_patch.py --training_mode normal

# Train a patch using the 'covert_style' mode with more realistic augmentations
python train_patch.py --training_mode covert_style --augmentations

# Resume a previous run from its checkpoint
python train_patch.py --training_mode normal --resume runs/<your-run-dir>/best_patch_checkpoint.pth

# Manually specify GPU IDs to use
python train_patch.py --training_mode normal --gpu_ids 0 1
```

**Key Arguments for `train_patch.py`:**

* `--config`: Path to your configuration file (default: `config.json`).
* `--training_mode`: The training strategy to use (`normal`, `covert_style`, `covert_procedural`).
* `--attack_mode`: The adversarial goal (`hide`, `misclassify`).
* `--decoy_class`: The target class ID for `misclassify` attacks.
* `--resume`: Path to a checkpoint to resume training.
* `--starter_image`: Path to an image to use as the starting point for the patch.
* `--gpu_ids`: Specify which GPU IDs to use (e.g., `--gpu_ids 0 1`).
* `--patches`: Number of patches to generate in sequence.
* `--scheduler`: Learning rate scheduler (`plateau`, `cosine_warm`).
* `--no-patience`: Disable early stopping.
* `--patch_coverage`: The desired patch coverage of the target object's area (e.g., `0.35` for 35%).
* `--augmentations`: Enable a more aggressive set of augmentations.
* `--batch_size`: Manually set the batch size and bypass autotuning.

---

### 2. Distributed Training (`train_patch_MG.py`)

This script is for high-performance training on machines with multiple GPUs. It **must** be launched with `torchrun`.

**Command Examples:**

```bash
# Launch training on all available GPUs
torchrun train_patch_MG.py

# Launch training on a specific number of GPUs (e.g., 4)
torchrun --nproc_per_node=4 train_patch_MG.py

# Launch with custom arguments
torchrun --nproc_per_node=8 train_patch_MG.py --max_epochs 500 --tv_weight 1e-5
```

> **Note**: This script uses a simplified set of arguments and gets most of its configuration from hardcoded defaults at the top of the file. Modify them as needed.

---

### 3. Hyperparameter Tuning (`automate_tuning.py`)

This script automates the process of finding the best noise parameters for a given training mode.

**How to Use:**

1.  Open the `automate_tuning.py` script.
2.  Modify the `PARAMETER_GRID` dictionary to define the ranges of `persistence`, `lacunarity`, `scale`, and `octaves` you want to test.
3.  Set the `TRAINING_MODE_TO_TEST` variable to the mode you want to optimize (e.g., `'normal'`).
4.  Set `EPOCHS_PER_RUN` to a low number (e.g., 80) for rapid testing.

**Run the script:**

```bash
python automate_tuning.py
```

The script will iterate through all parameter combinations, run a training and evaluation cycle for each, and print a ranked table of the results.

---

### 4. Evaluating a Patch (`evaluate_patch.py`)

After training, use this script to test your patch's effectiveness on the validation set.

**Evaluation Command:**

```bash
python evaluate_patch.py --model yolov11n.pt --patch runs/<your-run-directory>/best_patch.png
```

**Key Arguments for `evaluate_patch.py`:**

* `--model`: Path to the YOLO model to evaluate against.
* `--patch`: Path to the trained adversarial patch image.
* `--output`: Directory to save side-by-side comparison images.
* `--target_classes`: Specify which classes to attack during evaluation.
* `--coverage`: Set the patch coverage percentage for evaluation.
* `--conf`: Confidence threshold for detection.
* `--iou`: IoU threshold for NMS.

The script will output a summary table detailing the success rate and save visual comparisons to the output directory.
