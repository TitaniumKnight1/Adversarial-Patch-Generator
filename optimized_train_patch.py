# =================================================================================================
#           ADVERSARIAL PATCH TRAINING SCRIPT (V1.9)
# =================================================================================================
#
# This script trains an adversarial patch to deceive object detection models (YOLO).
# It includes multiple training modes to create patches that are either purely adversarial
# or designed to be visually covert by mimicking textures and patterns.
#
# Key Features:
# - Multi-Model Training: Train a single patch against an ensemble of YOLO models.
# - Multiple Training Modes:
#   - 'normal': Focuses on adversarial effectiveness with regularization.
#   - 'covert_style': Uses VGG-based style loss to mimic the texture of the surrounding environment.
#   - 'covert_procedural': Generates procedural camouflage based on Perlin noise.
# - Automatic Batch Size Tuning: Finds the optimal batch size for maximum GPU throughput.
# - Advanced Augmentations: Simulates real-world conditions like perspective shifts and color changes.
# - Rich CLI & Monitoring: Uses 'rich' for an interactive training dashboard and 'ntfy.sh' for remote notifications.
#
# =================================================================================================

# --- Core Libraries ---
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from ultralytics import YOLO
from PIL import Image
import os
import random
import time
import argparse
import torch.backends.cudnn as cudnn
from tensorboard import program
import traceback
import sys
import numpy as np
import torchvision.models as models
import psutil
import json
import requests
import math
import signal

# Set higher matmul precision for potential throughput gains on modern GPUs
try:
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass

# --- Third-Party Libraries ---
try:
    import noise  # For procedural camouflage generation
except ImportError:
    print("The 'noise' library is not installed. Please install it using: pip install noise")
    sys.exit(1)

from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts
from datetime import datetime

# --- Rich CLI Components for a clean UI ---
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.layout import Layout
from rich.panel import Panel

# --- Initialize Rich Console ---
console = Console()

# --- Notification Function ---
def send_notification(title, message, tags="tada"):
    """
    Sends a push notification to a specified ntfy.sh topic for remote monitoring.
    This allows you to get updates on your training progress on your phone or desktop.
    """
    try:
        # NOTE FOR USERS: Replace "PatchTraining" with your own private ntfy.sh topic URL.
        # For example: "https://ntfy.sh/your_secret_topic_name"
        requests.post(
            "https://ntfy.sh/EXAMPLE_TOPIC_CHANGE_ME",
            data=message.encode(encoding='utf-8'),
            headers={
                "Title": title.encode(encoding='utf-8'),
                "Tags": tags
            }
        )
    except Exception as e:
        console.print(f"⚠️ [yellow]Failed to send notification: {e}[/yellow]")


# --- Configuration Loading ---
def load_config(config_path):
    """
    Loads the main training configuration from the specified JSON file.
    The config file centralizes all key parameters for different training modes.
    """
    if not os.path.exists(config_path):
        console.print(f"💥 [bold red]Config file not found at '{config_path}'. Please create it.[/bold red]")
        sys.exit(1)
    with open(config_path, 'r') as f:
        config = json.load(f)
    console.print(f"✅ [green]Configuration loaded from '{config_path}'[/green]")
    return config

# --- Data Handling ---
def collate_fn(batch):
    """
    Custom collate function for the DataLoader. It filters out any `None` values
    that may occur if an image in a batch fails to load, preventing crashes.
    """
    batch = list(filter(lambda x: x is not None and x[0] is not None, batch))
    if not batch:
        return None, None, None, None
    images, box_label_pairs, patch_locations = zip(*batch)
    boxes, labels = zip(*box_label_pairs)
    return torch.stack(images, 0), boxes, labels, torch.stack(patch_locations, 0)

# --- Dataset Classes ---
class VisDroneDatasetLazy(Dataset):
    """
    A memory-efficient dataset class for VisDrone. It loads images and annotations
    from the disk "lazily" (i.e., on-the-fly as needed). This is ideal for systems
    with limited RAM where the entire dataset cannot be loaded at once.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_dir = os.path.join(root_dir, 'images')
        self.annotation_dir = os.path.join(root_dir, 'annotations_v11')
        self.image_files = sorted(os.listdir(self.image_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
            original_size = image.size
        except Exception:
            # Return None if an image is corrupted to be filtered by collate_fn
            return None, None, None

        boxes, labels = [], []
        annotation_name = os.path.splitext(img_name)[0] + '.txt'
        annotation_path = os.path.join(self.annotation_dir, annotation_name)
        if os.path.exists(annotation_path):
            with open(annotation_path, 'r') as f:
                for line in f.readlines():
                    try:
                        parts = line.strip().split(',')
                        if len(parts) >= 6:
                            x1, y1, w, h = map(float, parts[:4])
                            class_id = int(parts[5])
                            boxes.append([x1, y1, x1 + w, y1 + h])
                            labels.append(class_id)
                    except (ValueError, IndexError):
                        continue
        
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        if self.transform:
            image = self.transform(image)
        
        # Scale bounding boxes to match the transformed image size (e.g., 640x640)
        if boxes_tensor.nelement() > 0:
            scale_x, scale_y = 640 / original_size[0], 640 / original_size[1]
            boxes_tensor[:, [0, 2]] *= scale_x
            boxes_tensor[:, [1, 3]] *= scale_y
        
        patch_location = torch.zeros(4) # Placeholder, not used in this script
        return image, (boxes_tensor, labels_tensor), patch_location

class VisDroneDatasetPreload(Dataset):
    """
    A high-performance dataset class that pre-loads and pre-processes the entire
    dataset into RAM. This significantly speeds up training by eliminating disk I/O
    bottlenecks, but requires a large amount of available RAM. It also caches the
    processed data to disk for fast subsequent initializations.
    """
    def __init__(self, root_dir, transform=None, cache_file="preprocessed_dataset.pth"):
        cache_path = os.path.join(root_dir, cache_file)
        if os.path.exists(cache_path):
            console.print(f"⏳ [blue]Loading pre-processed dataset from cache: {cache_path}[/blue]")
            try:
                cached_data = torch.load(cache_path, weights_only=False)
                # Verify cache format to handle updates to the preprocessing logic
                if 'annotations' not in cached_data or not isinstance(cached_data['annotations'][0], tuple):
                    console.print(f"⚠️ [yellow]Cache format is outdated. Forcing regeneration.[/yellow]")
                    raise TypeError("Old cache format")

                self.images = cached_data['images']
                self.annotations = cached_data['annotations']
                console.print(f"✅ [green]Cached dataset loaded successfully. {len(self.images)} items in memory.[/green]")
                return
            except Exception as e:
                console.print(f"⚠️ [yellow]Could not load cache or cache is invalid: {e}. Re-processing dataset.[/yellow]")

        console.print(f"⏳ [magenta]No cache found or cache invalid. Pre-processing dataset into tensors...[/magenta]")
        self.images, self.annotations = [], []
        image_dir, annotation_dir = os.path.join(root_dir, 'images'), os.path.join(root_dir, 'annotations_v11')
        image_files = sorted(os.listdir(image_dir))

        with Progress(console=console) as progress:
            task = progress.add_task("[cyan]Processing Data...", total=len(image_files))
            for img_name in image_files:
                progress.update(task, advance=1)
                img_path = os.path.join(image_dir, img_name)
                try:
                    with Image.open(img_path) as img:
                        img_rgb = img.convert("RGB")
                        original_size = img_rgb.size
                        self.images.append(transform(img_rgb) if transform else TF.to_tensor(img_rgb))
                except Exception as e:
                    console.print(f"Warning: Could not load image {img_path}. Skipping. Error: {e}", style="yellow")
                    continue

                boxes, labels = [], []
                annotation_name = f"{os.path.splitext(img_name)[0]}.txt"
                annotation_path = os.path.join(annotation_dir, annotation_name)
                if os.path.exists(annotation_path):
                    with open(annotation_path, 'r') as f:
                        for line in f.readlines():
                            try:
                                parts = line.strip().split(',')
                                if len(parts) >= 6:
                                    x1, y1, w, h = map(float, parts[:4])
                                    class_id = int(parts[5])
                                    boxes.append([x1, y1, x1 + w, y1 + h])
                                    labels.append(class_id)
                            except (ValueError, IndexError):
                                continue
                boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
                labels_tensor = torch.tensor(labels, dtype=torch.long)

                if boxes_tensor.nelement() > 0:
                    scale_x, scale_y = 640 / original_size[0], 640 / original_size[1]
                    boxes_tensor[:, [0, 2]] *= scale_x
                    boxes_tensor[:, [1, 3]] *= scale_y
                self.annotations.append((boxes_tensor, labels_tensor))
        
        console.print(f"💾 [blue]Saving pre-processed dataset to cache for future runs...[/blue]")
        try:
            torch.save({'images': self.images, 'annotations': self.annotations}, cache_path)
            console.print(f"✅ [green]Dataset cached successfully at: {cache_path}[/green]")
        except Exception as e:
            console.print(f"⚠️ [red]Could not save cache file: {e}[/red]")

    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        patch_location = torch.zeros(4)
        boxes, labels = self.annotations[idx]
        return self.images[idx], (boxes, labels), patch_location

class DummyDataset(Dataset):
    """
    A lightweight dataset that generates random tensors on the fly.
    Used exclusively for the batch size autotuning process to quickly find
    the VRAM limit without loading any real data from disk.
    """
    def __init__(self, length=2048, image_size=(3, 640, 640)):
        self.length, self.image_size = length, image_size
    def __len__(self): return self.length
    def __getitem__(self, idx): return torch.rand(self.image_size)

# --- Loss Functions and Patch Regularizers ---

class TotalVariationLoss(torch.nn.Module):
    """
    Computes the Total Variation (TV) loss, a regularizer that encourages spatial
    smoothness in the generated patch. It penalizes large differences between
    adjacent pixel values, reducing noise and creating more coherent patterns.
    """
    def forward(self, patch):
        if patch.dim() == 3: patch = patch.unsqueeze(0)
        wh_diff = torch.sum(torch.abs(patch[:, :, :, 1:] - patch[:, :, :, :-1]))
        ww_diff = torch.sum(torch.abs(patch[:, :, 1:, :] - patch[:, :, :-1, :]))
        return (wh_diff + ww_diff) / (patch.size(2) * patch.size(3))

class NonPrintabilityScore(torch.nn.Module):
    """
    Computes the Non-Printability Score (NPS) loss, which penalizes colors that
    are difficult to reproduce accurately with physical printers. This loss helps
    in creating patches that are more robust when transferred to the real world.
    It measures the minimum distance of each pixel's color to a predefined set
    of 32 common printable colors.
    """
    def __init__(self, device):
        super(NonPrintabilityScore, self).__init__()
        # A small, representative set of 32 printable colors (e.g., from a standard inkjet printer palette)
        printable_colors_hex = [
            '#FFFFFF', '#000000', '#C2C2C2', '#7F7F7F', '#FF0000', '#800000', '#FFFF00', '#808000',
            '#00FF00', '#008000', '#00FFFF', '#008080', '#0000FF', '#000080', '#FF00FF', '#800080',
            '#FFC90E', '#E87400', '#A83A00', '#732900', '#7030A0', '#4F2370', '#0070C0', '#004C80',
            '#00B050', '#007838', '#C00000', '#8C0000', '#FFD966', '#F4B183', '#D6DCE4', '#ADB9CA'
        ]
        
        printable_colors_rgb = torch.tensor(
            [[int(h[i:i+2], 16) / 255.0 for i in (1, 3, 5)] for h in printable_colors_hex],
            device=device
        )
        self.printable_colors = printable_colors_rgb.unsqueeze(0).unsqueeze(2).unsqueeze(3)

    def forward(self, patch):
        patch_flat = patch.view(1, 3, -1)
        colors_flat = self.printable_colors.view(-1, 3).T.unsqueeze(0)
        color_diff = patch_flat.unsqueeze(3) - colors_flat.unsqueeze(2)
        color_dist_sq = torch.sum(color_diff ** 2, dim=1)
        min_dist_sq, _ = torch.min(color_dist_sq, dim=2)
        return torch.mean(min_dist_sq)


# --- VGG Feature Extractor for Style-Based Camouflage ---
class VGGFeatureExtractor(torch.nn.Module):
    """
    Extracts features from intermediate layers of a pre-trained VGG19 network.
    These features are used to compute the style loss. The network is set to
    evaluation mode and its parameters are frozen to act as a fixed feature extractor.
    """
    def __init__(self, device):
        super(VGGFeatureExtractor, self).__init__()
        vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()
        for param in vgg19.parameters():
            param.requires_grad = False
        
        # Using shallower layers (up to relu2_1) is faster and often sufficient for texture
        self.feature_layers = vgg19[:7] 
        self.layer_indices = {1, 6} # Indices for relu1_1, relu2_1
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x):
        features = []
        x = self.normalize(x)
        for i, layer in enumerate(self.feature_layers):
            x = layer(x)
            if i in self.layer_indices:
                features.append(x)
        return features

# --- Style-Based Loss ---
class StyleLoss(torch.nn.Module):
    """
    Computes a style loss that encourages the patch to adopt the texture of a
    target image. It does this by comparing the Gram matrices of their VGG features.
    The Gram matrix captures the correlations between different feature channels,
    representing the texture and style of an image.
    """
    def __init__(self, device):
        super(StyleLoss, self).__init__()
        self.feature_extractor = VGGFeatureExtractor(device)
        self.l1_loss = torch.nn.L1Loss()

    def gram_matrix(self, features):
        b, c, h, w = features.size()
        features_flat = features.view(b, c, h * w)
        gram = torch.bmm(features_flat, features_flat.transpose(1, 2))
        return gram.div(c * h * w)

    def get_gram_matrices(self, image_batch):
        """Helper function to compute Gram matrices for a batch of images."""
        features = self.feature_extractor(image_batch)
        gram_matrices = [self.gram_matrix(f) for f in features]
        return gram_matrices

    def forward(self, source, target):
        # Handle different batch sizes between source (patch) and target (images)
        if source.dim() == 3 and target.dim() == 4:
            source = source.unsqueeze(0).expand(target.size(0), -1, -1, -1)
        elif source.dim() == 3 and target.dim() == 3:
            source = source.unsqueeze(0)
            target = target.unsqueeze(0)

        source_features = self.feature_extractor(source)
        target_features = self.feature_extractor(target)

        style_loss = 0.0
        for sf, tf in zip(source_features, target_features):
            gram_source = self.gram_matrix(sf)
            gram_target = self.gram_matrix(tf)
            style_loss += self.l1_loss(gram_source, gram_target)
        return style_loss

# --- Procedural Camouflage Generation ---
def generate_camouflage_pattern(width, height, colors, scale=25.0, octaves=4, persistence=0.6, lacunarity=2.0, device='cpu'):
    """
    Generates a procedural camouflage pattern using Perlin noise. This creates a
    natural-looking, random texture based on a provided color palette.
    """
    num_colors = colors.shape[0]
    world = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            world[i][j] = noise.pnoise2(i / scale, j / scale, octaves=octaves, persistence=persistence, lacunarity=lacunarity, repeatx=width, repeaty=height, base=random.randint(0, 100))
    
    # Normalize noise values and map them to the color palette
    world_normalized = (world - np.min(world)) / (np.max(world) - np.min(world))
    color_indices = np.floor(world_normalized * num_colors).astype(np.int32)
    color_indices[color_indices == num_colors] = num_colors - 1
    
    camo_pattern_np = colors[color_indices].cpu().numpy()
    camo_pattern_tensor = torch.from_numpy(camo_pattern_np).permute(2, 0, 1).to(device)
    
    # Apply a slight blur for a smoother, more natural appearance
    blurred_pattern = TF.gaussian_blur(camo_pattern_tensor.unsqueeze(0), kernel_size=[3, 3]).squeeze(0)
    return blurred_pattern

# --- Autotuning and Training ---

def find_optimal_batch_size(model_to_tune, device, dataset_len, initial_batch_size=8, num_benchmark_steps=50):
    """
    Performs a two-phase search to find the batch size with the highest throughput (images/sec).
    This is more effective than just finding the max size, as smaller batch sizes can sometimes
    be faster due to better cache utilization.
    """
    console.print(Panel("[bold yellow]🚀 Starting Throughput-Based Batch Size Autotune[/bold yellow]", 
                        title="[cyan]Optimal Batch Finder[/cyan]", border_style="yellow"))

    # --- Phase 1: Binary search to find the absolute maximum batch size that fits in VRAM. ---
    low = 1
    high = 256 
    max_workable_size = 0
    
    with console.status("[bold blue]Phase 1/2: Finding memory ceiling...") as status:
        while low <= high:
            mid = (low + high) // 2
            if mid == 0: low = 1; continue
            status.update(f"[bold blue]Phase 1/2: Probing batch size {mid}...[/bold blue]")
            try:
                dummy_data = DummyDataset(length=mid)
                dataloader = DataLoader(dummy_data, batch_size=mid, pin_memory=True)
                images = next(iter(dataloader)).to(device)
                images.requires_grad_(True)
                
                with torch.amp.autocast(device.type):
                    output = model_to_tune.model(images)
                    dummy_loss = output[0].sum()
                dummy_loss.backward()
                
                max_workable_size = mid
                low = mid + 1
                del images, dataloader, output, dummy_loss
                torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                high = mid - 1
                torch.cuda.empty_cache()
            except StopIteration:
                max_workable_size = mid; break
    
    if max_workable_size == 0:
        console.print("💥 [bold red]Could not fit even a batch size of 1 in memory. Aborting.[/bold red]")
        sys.exit(1)
        
    console.print(f"✅ [green]Phase 1 Complete: Max workable batch size is {max_workable_size}.[/green]")

    # --- Phase 2: Profile a range of batch sizes to find the one that is fastest in practice. ---
    best_batch_size = 0
    max_throughput = 0.0
    
    search_space = sorted(list(set([2**i for i in range(3, 11) if 2**i <= max_workable_size] + [max_workable_size])))
    
    console.print(f"🔬 [cyan]Phase 2/2: Benchmarking throughput for batch sizes: {search_space}[/cyan]")
    
    with Progress(console=console) as progress:
        task = progress.add_task("[cyan]Profiling...", total=len(search_space))
        for size in search_space:
            progress.update(task, description=f"[cyan]Testing size {size}...[/cyan]")
            try:
                dummy_data = DummyDataset(length=size * num_benchmark_steps)
                dataloader = DataLoader(dummy_data, batch_size=size, pin_memory=True)
                
                # Warm-up iterations
                for _ in range(5):
                    images = next(iter(dataloader)).to(device)
                    images.requires_grad_(True)
                    with torch.amp.autocast(device.type):
                        output = model_to_tune.model(images)
                        dummy_loss = output[0].sum()
                    dummy_loss.backward()
                torch.cuda.empty_cache()
                
                # Benchmark
                start_time = time.time()
                if device.type == 'cuda': torch.cuda.synchronize()

                for i, images in enumerate(dataloader):
                    if i >= num_benchmark_steps: break
                    images = images.to(device, non_blocking=True)
                    images.requires_grad_(True)
                    with torch.amp.autocast(device.type):
                        output = model_to_tune.model(images)
                        dummy_loss = output[0].sum()
                    dummy_loss.backward()

                if device.type == 'cuda': torch.cuda.synchronize()
                end_time = time.time()
                
                duration = end_time - start_time
                throughput = (num_benchmark_steps * size) / duration
                
                console.print(f"  - Size [b]{size}[/b]: {throughput:.2f} images/sec")
                
                if throughput > max_throughput:
                    max_throughput = throughput
                    best_batch_size = size
                
                del images, dataloader, output, dummy_loss
                torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                console.print(f"  - Size [b]{size}[/b]: [red]Out of Memory during benchmark.[/red]")
                torch.cuda.empty_cache()
                break 
            
            progress.update(task, advance=1)

    final_size = best_batch_size
    console.print(Panel(f"🏁 [bold green]Autotune Complete![/bold green]\n"
                        f"   - Max Throughput: [b]{max_throughput:.2f} images/sec[/b]\n"
                        f"   - Optimal Batch Size: [b]{final_size}[/b]",
                        title="[cyan]Optimal Batch Found[/cyan]", border_style="green"))
    return max(1, final_size)

# --- Main Training Loop ---

# Global state for signal handler to allow graceful exit on Ctrl+C
training_state = {'best_patch': None, 'log_dir': None, 'should_exit': False}

def signal_handler(sig, frame):
    """
    Handles Ctrl+C interrupts. It sets a flag to stop the training loop cleanly
    and saves the best patch found so far before exiting.
    """
    console.print("\n[bold yellow]Ctrl+C detected. Saving best patch and exiting gracefully...[/bold yellow]")
    if training_state['best_patch'] is not None and training_state['log_dir'] is not None:
        save_path = os.path.join(training_state['log_dir'], "best_patch_interrupt.png")
        try:
            T.ToPILImage()(training_state['best_patch'].cpu()).save(save_path)
            console.print(f"✅ [green]Best patch saved to: {save_path}[/green]")
        except Exception as e:
            console.print(f"💥 [red]Could not save best patch: {e}[/red]")
    training_state['should_exit'] = True

def train_adversarial_patch(config, models, log_dir, device, dataloader, num_workers, pin_memory, args, resume_path=None, starter_image_path=None):
    """
    The main training loop for generating the adversarial patch.
    """
    if device.type == 'cuda': cudnn.benchmark = True
    writer = SummaryWriter(log_dir=log_dir)
    
    # Extract parameters from the active config
    hp, lw = config['hyperparameters'], config['loss_weights']
    patch_size, learning_rate = args.patch_size, hp['base_learning_rate']
    early_stopping_patience = hp['early_stopping_patience']
    
    # Load all loss weights; unused ones will default to 0.0
    adv_weight = lw.get('adv_weight', 100.0)
    tv_weight = lw.get('tv_weight', 0.0)
    nps_weight = lw.get('nps_weight', 0.0)
    style_weight = lw.get('style_weight', 0.0)
    pattern_weight = lw.get('pattern_weight', 0.0)

    max_epochs, target_classes = hp['max_epochs'], args.target_classes

    # Initialize the patch (either from a starter image or random noise)
    if starter_image_path and os.path.exists(starter_image_path):
        console.print(f"🌱 [cyan]Initializing patch from starter image: {starter_image_path}[/cyan]")
        starter_image = Image.open(starter_image_path).convert("RGB")
        transform_starter = T.Compose([T.Resize((patch_size, patch_size)), T.ToTensor()])
        adversarial_patch = transform_starter(starter_image).to(device, non_blocking=True)
        adversarial_patch.requires_grad_(True)
    else:
        console.print(f"🎨 [cyan]Initializing patch with random noise.[/cyan]")
        adversarial_patch = torch.rand((3, patch_size, patch_size), device=device, requires_grad=True)

    # Prefer channels_last only for 4D tensors; patch is 3D so skip
    adversarial_patch = adversarial_patch  # keep default memory format

    # Setup optimizer, scaler (for mixed precision), and scheduler
    try:
        optimizer = torch.optim.AdamW([adversarial_patch], lr=learning_rate, amsgrad=True, fused=(device.type == 'cuda'))
    except TypeError:
        optimizer = torch.optim.AdamW([adversarial_patch], lr=learning_rate, amsgrad=True)
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))
    
    if args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=hp['plateau_patience'])
    else: # Default to cosine_warm
        restart_epochs = hp.get('cosine_restart_epochs', max(1, max_epochs // 4))
        console.print(f"🔥 [bold cyan]Using CosineAnnealingWarmRestarts with T_0 = {restart_epochs} epochs.[/bold cyan]")
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=restart_epochs, T_mult=1, eta_min=1e-6)

    # Initialize all required loss functions based on weights
    total_variation = TotalVariationLoss().to(device)
    nps_loss_fn = NonPrintabilityScore(device).to(device)
    style_loss_fn, pattern_loss_fn = None, None
    
    if style_weight > 0:
        style_loss_fn = StyleLoss(device).to(device)
    if pattern_weight > 0:
        pattern_loss_fn = torch.nn.L1Loss().to(device)

    console.print(f"🎨 [bold blue]Training Mode: {args.training_mode.replace('_', ' ').title()}[/bold blue]")

    # Precompute set for faster membership tests
    if target_classes:
        target_classes_set = set(int(c) for c in target_classes)
    else:
        target_classes_set = None

    # Pre-compute style target for 'normal' mode if style_weight is active
    target_style_image = None
    target_style_grams = None
    noise_params = config.get('noise_parameters', {})
    if args.training_mode == 'normal' and style_weight > 0:
        console.print("🎨 [cyan]Generating fixed Perlin noise style target for patch regularization.[/cyan]")
        random_colors = torch.rand(4, 3, device=device)
        target_style_image = generate_camouflage_pattern(
            patch_size, patch_size, random_colors, device=device,
            scale=noise_params.get('scale', 50.0),
            octaves=noise_params.get('octaves', 6),
            persistence=noise_params.get('persistence', 0.5),
            lacunarity=noise_params.get('lacunarity', 2.0)
        )
        writer.add_image('Target Style Pattern', target_style_image, 0)
        
        console.print("✅ [green]Style loss optimization enabled: Pre-computing Gram matrices for fixed style target.[/green]")
        with torch.no_grad():
            target_style_image_fp32 = target_style_image.float().unsqueeze(0)
            target_style_grams = style_loss_fn.get_gram_matrices(target_style_image_fp32)

    # Resume training from checkpoint if provided
    start_epoch = 0
    if resume_path and os.path.exists(resume_path):
        console.print(f"🔄 [blue]Resuming training from checkpoint: {resume_path}[/blue]")
        checkpoint = torch.load(resume_path)
        adversarial_patch.data = checkpoint['patch_state_dict'].to(device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scaler_state_dict' in checkpoint: scaler.load_state_dict(checkpoint['scaler_state_dict'])
        if 'scheduler_state_dict' in checkpoint and args.scheduler == checkpoint.get('scheduler_type'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        console.print(f"Resumed from epoch {start_epoch}.")

    # Define augmentation pipelines
    adv_patch_transform_standard = T.Compose([
        T.RandomPerspective(distortion_scale=0.6, p=0.8, fill=0),
        T.RandomRotation(degrees=45, fill=0),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0)),
        lambda x: torch.clamp(x + torch.randn_like(x) * 0.1, 0, 1)
    ])

    adv_patch_transform_real = T.Compose([
        T.RandomPerspective(distortion_scale=0.8, p=0.9, fill=0),
        T.RandomRotation(degrees=90, fill=0),
        T.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.3),
        T.GaussianBlur(kernel_size=(9, 13), sigma=(0.1, 7.0)),
        T.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
        lambda x: torch.clamp(x + torch.randn_like(x) * 0.15, 0, 1)
    ])

    adv_patch_transform = adv_patch_transform_real if args.augmentations else adv_patch_transform_standard

    # --- Rich UI Setup ---
    model_names_str = ", ".join([os.path.basename(m.ckpt_path) for m in models])
    target_classes_str = "All" if not target_classes else str(target_classes)
    layout = Layout()
    config_panel_content = (f"🚀 [bold magenta]Starting Adversarial Patch Training[/bold magenta]\n"
                          f"   - [b]Training Mode[/b]: [yellow]{args.training_mode.replace('_', ' ').title()}[/yellow]\n"
                          f"   - [b]Attack Strategy[/b]: [yellow]{args.attack_mode.capitalize()}[/yellow]"
                          f"{f' (Decoy Class: {args.decoy_class})' if args.decoy_class is not None else ''}\n"
                          f"   - [b]Augmentations[/b]: {'[bold green]Enabled[/bold green]' if args.augmentations else '[bold red]Disabled[/bold red]'}\n"
                          f"   - [b]Target Models[/b]: [cyan]{model_names_str}[/cyan]\n"
                          f"   - [b]Target Classes[/b]: [cyan]{target_classes_str}[/cyan]\n"
                          f"   - [b]Device[/b]: [cyan]{device.type.upper()}[/cyan]\n"
                          f"   - [b]Batch Size[/b]: [cyan]{hp['base_batch_size']}[/cyan]\n"
                          f"   - [b]Patch Coverage[/b]: [cyan]{args.patch_coverage:.0%}[/cyan]\n"
                          f"   - [b]Scheduler[/b]: [cyan]{args.scheduler}[/cyan]\n"
                          f"   - [b]Learning Rate[/b]: [cyan]{learning_rate:.2e}[/cyan]\n"
                          f"   - [b]Loss Weights[/b]: Adv: {adv_weight:.1e}")
    if tv_weight > 0: config_panel_content += f", TV: {tv_weight:.1e}"
    if nps_weight > 0: config_panel_content += f", NPS: {nps_weight:.1e}"
    if style_weight > 0: config_panel_content += f", Style: {style_weight:.1e}"
    if pattern_weight > 0: config_panel_content += f", Pattern: {pattern_weight:.1e}"
    if noise_params:
        config_panel_content += (f"\n   - [b]Noise Params[/b]: Scale: {noise_params.get('scale', 50.0)}, Octaves: {noise_params.get('octaves', 6)}, "
                               f"Pers: {noise_params.get('persistence', 0.5)}, Lac: {noise_params.get('lacunarity', 2.0)}")
    
    layout.split_column(
        Layout(Panel(config_panel_content, title="[yellow]Training Configuration[/yellow]", border_style="yellow"), name="config"),
        Layout(name="progress", size=3),
        Layout(name="table", ratio=1)
    )
    progress = Progress(TextColumn("[bold blue]{task.description}", justify="right"), BarColumn(bar_width=None), "[progress.percentage]{task.percentage:>3.1f}%", "•", MofNCompleteColumn(), "•", TimeRemainingColumn(), console=console)
    layout["progress"].update(Panel(progress, title="[yellow]Current Epoch Progress[/yellow]", border_style="yellow"))

    best_loss, epochs_no_improve, best_loss_epoch = float('inf'), 0, -1
    epoch_results, stop_reason, final_epoch = [], "Max epochs reached", 0
    live = Live(layout, screen=True, redirect_stderr=False, vertical_overflow="visible", console=console)
    
    # --- Epoch Loop ---
    try:
        with live:
            for epoch in range(start_epoch, max_epochs):
                if training_state['should_exit']: stop_reason = "Manually interrupted (Ctrl+C)"; break
                final_epoch = epoch
                epoch_start_time = time.time()
                total_adv_loss, total_tv_loss, total_nps_loss, total_style_loss, total_pattern_loss = 0, 0, 0, 0, 0
                task_id = progress.add_task(f"Epoch {epoch + 1}", total=len(dataloader))

                # Prepare utilities reused across the epoch
                target_camo_pattern = None  # lazily initialized on first batch if needed
                target_textures_buffer = None  # reused buffer for covert_style targets

                # --- Batch Loop ---
                for i, batch_data in enumerate(dataloader):
                    if training_state['should_exit']: break
                    if batch_data is None or batch_data[0] is None:
                        # Smooth progress: advance by 1 for each skipped batch
                        progress.update(task_id, advance=1)
                        continue
                    
                    images, gt_boxes_batch, gt_labels_batch, _ = batch_data
                    # Move to device (keep standard contiguous layout) and enforce float32
                    images = images.to(device, non_blocking=pin_memory, dtype=torch.float32)

                    # Lazily generate procedural camo target from first batch (avoids double dataloader pass)
                    if target_camo_pattern is None and args.training_mode == 'covert_procedural' and pattern_weight > 0:
                        try:
                            first_images = images  # already on device
                            first_img_context = first_images[0].permute(1, 2, 0)
                            pixels = first_img_context.reshape(-1, 3)
                            sample_size = min(pixels.shape[0], 1000)
                            palette_pool_indices = torch.randperm(pixels.shape[0], device=pixels.device)[:sample_size]
                            palette_pool = pixels[palette_pool_indices]
                            final_palette_indices = torch.randperm(palette_pool.shape[0], device=pixels.device)[:4]
                            color_palette = palette_pool[final_palette_indices]
                            target_camo_pattern = generate_camouflage_pattern(
                                patch_size, patch_size, color_palette, device=device,
                                scale=noise_params.get('scale', 25.0),
                                octaves=noise_params.get('octaves', 4),
                                persistence=noise_params.get('persistence', 0.6),
                                lacunarity=noise_params.get('lacunarity', 2.0)
                            )
                        except Exception as _e:
                            pass
                    
                    # Prepare a tensor to hold style targets for each image in the batch (reuse buffer)
                    target_textures_batch = None
                    if args.training_mode == 'covert_style':
                        if target_textures_buffer is None or target_textures_buffer.size(0) != images.size(0) or target_textures_buffer.size(2) != patch_size:
                            target_textures_buffer = torch.empty((images.size(0), 3, patch_size, patch_size), device=device)
                        target_textures_buffer.zero_()
                        target_textures_batch = target_textures_buffer
                    
                    # Augment the patch for this batch
                    augmented_patch = adv_patch_transform(adversarial_patch)
                    augmented_patch.data.clamp_(0,1)

                    # Cache resized patches per-batch to avoid repeated resizes for identical sizes
                    resized_cache = {}

                    # --- Apply Patch to Batch ---
                    for img_idx in range(images.size(0)):
                        gt_boxes, gt_labels = gt_boxes_batch[img_idx], gt_labels_batch[img_idx]
                        targetable_indices = [j for j, label in enumerate(gt_labels) if (int(label.item()) in target_classes_set)] if target_classes_set else list(range(len(gt_boxes)))

                        if targetable_indices:
                            # If targets are present, apply patch to a random one
                            selected_idx = random.choice(targetable_indices)
                            box = gt_boxes[selected_idx].int()
                            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                            target_w, target_h = x2 - x1, y2 - y1
                            if target_w <=0 or target_h <=0: continue
                            
                            patch_area = target_w * target_h * args.patch_coverage
                            patch_size_to_apply = max(20, min(int(math.sqrt(patch_area)), min(images.shape[2], images.shape[3])))
                            
                            # Resize once per unique size in this batch
                            resized_patch = resized_cache.get(patch_size_to_apply)
                            if resized_patch is None:
                                try:
                                    resized_patch = TF.resize(augmented_patch.unsqueeze(0), (patch_size_to_apply, patch_size_to_apply), antialias=True).squeeze(0)
                                    resized_cache[patch_size_to_apply] = resized_patch
                                except Exception:
                                    continue
                            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                            patch_x = max(0, min(center_x - patch_size_to_apply // 2, images.shape[3] - patch_size_to_apply))
                            patch_y = max(0, min(center_y - patch_size_to_apply // 2, images.shape[2] - patch_size_to_apply))
                            
                            # For style mode, extract the texture from around the patch location BEFORE applying
                            if args.training_mode == 'covert_style':
                                surrounding_x1, surrounding_y1 = max(0, patch_x - 15), max(0, patch_y - 15)
                                surrounding_x2, surrounding_y2 = min(images.shape[3], patch_x + patch_size_to_apply + 15), min(images.shape[2], patch_y + patch_size_to_apply + 15)
                                surrounding_area = images[img_idx, :, surrounding_y1:surrounding_y2, surrounding_x1:surrounding_x2]
                                if surrounding_area.nelement() > 0:
                                    resized_texture = TF.resize(surrounding_area.unsqueeze(0), (patch_size, patch_size), antialias=True).squeeze(0)
                                    target_textures_batch[img_idx] = resized_texture

                            images[img_idx, :, patch_y:patch_y+patch_size_to_apply, patch_x:patch_x+patch_size_to_apply] = resized_patch
                        else:
                            # If no targets, apply patch to a random location (also cache exact-size)
                            ps = patch_size
                            resized_patch = resized_cache.get(ps)
                            if resized_patch is None:
                                resized_patch = augmented_patch  # already correct size
                                resized_cache[ps] = resized_patch
                            patch_x, patch_y = random.randint(0, images.shape[3] - ps), random.randint(0, images.shape[2] - ps)
                            if args.training_mode == 'covert_style':
                                texture_crop_area = images[img_idx, :, patch_y:patch_y+ps, patch_x:patch_x+ps]
                                target_textures_batch[img_idx] = texture_crop_area.clone()
                            images[img_idx, :, patch_y:patch_y+ps, patch_x:patch_x+ps] = resized_patch

                    optimizer.zero_grad(set_to_none=True)
                    current_model = random.choice(models)

                    # --- Loss Calculation ---
                    style_loss = torch.tensor(0.0, device=device)
                    if style_weight > 0 and style_loss_fn is not None:
                        patch_fp32 = adversarial_patch.float()
                        if args.training_mode == 'normal':
                            # Optimized style loss using pre-computed Gram matrices
                            patch_grams = style_loss_fn.get_gram_matrices(patch_fp32.unsqueeze(0))
                            current_style_loss = 0.0
                            for patch_g, target_g in zip(patch_grams, target_style_grams):
                                current_style_loss += style_loss_fn.l1_loss(patch_g, target_g)
                            style_loss = current_style_loss
                        elif args.training_mode == 'covert_style':
                            style_target_fp32 = target_textures_batch.float()
                            style_loss = style_loss_fn(patch_fp32, style_target_fp32)

                    # Use Automatic Mixed Precision for performance
                    with torch.amp.autocast(device.type):
                        raw_preds = current_model.model(images)[0].transpose(1, 2)
                        objectness_scores = raw_preds[..., 4]
                        class_probs = raw_preds[..., 5:]
                        
                        # Select adversarial loss based on attack mode
                        if args.attack_mode == 'hide': 
                            adversarial_loss = objectness_scores.mean()
                        elif args.attack_mode == 'misclassify': 
                            adversarial_loss = -class_probs[..., args.decoy_class].mean()
                        else: 
                            adversarial_loss = torch.tensor(0.0, device=device)

                        # Calculate regularization losses
                        tv_loss = total_variation(adversarial_patch) if tv_weight > 0 else torch.tensor(0.0, device=device)
                        nps_loss = nps_loss_fn(adversarial_patch) if nps_weight > 0 else torch.tensor(0.0, device=device)
                        
                        pattern_loss = torch.tensor(0.0, device=device)
                        if pattern_weight > 0 and pattern_loss_fn is not None and target_camo_pattern is not None:
                            pattern_loss = pattern_loss_fn(adversarial_patch, target_camo_pattern)
                            
                        # Combine all losses with their respective weights
                        total_loss = (adv_weight * adversarial_loss) + (tv_weight * tv_loss) + (nps_weight * nps_loss) + (style_weight * style_loss) + (pattern_weight * pattern_loss)

                    # --- Backpropagation ---
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    adversarial_patch.data.clamp_(0, 1) # Ensure patch values are valid
                    
                    # Accumulate loss values for logging
                    total_adv_loss += adversarial_loss.item()
                    if tv_weight > 0: total_tv_loss += tv_loss.item()
                    if nps_weight > 0: total_nps_loss += nps_loss.item()
                    if style_weight > 0: total_style_loss += style_loss.item()
                    if pattern_weight > 0: total_pattern_loss += pattern_loss.item()

                    # Smooth progress: advance by 1 each batch
                    progress.update(task_id, advance=1)

                # --- End of Epoch ---
                progress.remove_task(task_id)
                avg_adv_loss = total_adv_loss / len(dataloader)
                avg_tv_loss = total_tv_loss / len(dataloader) if tv_weight > 0 else 0
                avg_nps_loss = total_nps_loss / len(dataloader) if nps_weight > 0 else 0
                avg_style_loss = total_style_loss / len(dataloader) if style_weight > 0 else 0
                avg_pattern_loss = total_pattern_loss / len(dataloader) if pattern_weight > 0 else 0
                avg_total_loss = (adv_weight * avg_adv_loss) + (tv_weight * avg_tv_loss) + (nps_weight * avg_nps_loss) + (style_weight * avg_style_loss) + (pattern_weight * avg_pattern_loss)
                epoch_duration = time.time() - epoch_start_time
                current_lr = optimizer.param_groups[0]['lr']
                
                # --- Checkpointing and Early Stopping ---
                if avg_total_loss < best_loss:
                    best_loss, epochs_no_improve, best_loss_epoch = avg_total_loss, 0, epoch + 1
                    training_state['best_patch'] = adversarial_patch.data.clone()
                    training_state['log_dir'] = log_dir
                    torch.save({'epoch': epoch + 1, 'patch_state_dict': adversarial_patch.data.clone(), 'optimizer_state_dict': optimizer.state_dict(), 'scaler_state_dict': scaler.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'scheduler_type': args.scheduler}, os.path.join(log_dir, "best_patch_checkpoint.pth"))
                    T.ToPILImage()(adversarial_patch.cpu()).save(os.path.join(log_dir, "best_patch.png"))
                else:
                    epochs_no_improve += 1
                
                # --- Dynamic UI Update ---
                result_row = {"epoch": epoch + 1, "duration": f"{epoch_duration:.1f}", "adv_loss": avg_adv_loss}
                if tv_weight > 0: result_row["tv_loss"] = avg_tv_loss
                if nps_weight > 0: result_row["nps_loss"] = avg_nps_loss
                if style_weight > 0: result_row["style_loss"] = avg_style_loss
                if pattern_weight > 0: result_row["pattern_loss"] = avg_pattern_loss
                patience_str = f"{epochs_no_improve}/{early_stopping_patience}" if not args.no_patience else f"{epochs_no_improve}"
                result_row.update({"total_loss": avg_total_loss, "lr": current_lr, "patience": patience_str})
                epoch_results.append(result_row)
                
                results_table = Table(title="Epoch Results", expand=True, border_style="blue")
                header = ["Epoch", "Time (s)", "Adv Loss"]
                if tv_weight > 0: header.append("TV Loss")
                if nps_weight > 0: header.append("NPS Loss")
                if style_weight > 0: header.append("Style Loss")
                if pattern_weight > 0: header.append("Pattern Loss")
                header.extend(["Total Loss", "LR", "Patience"])
                for col in header: results_table.add_column(col, no_wrap=True)
                
                for result in epoch_results[-20:]:
                    row_values = [f"{v:.4e}" if isinstance(v, float) and 'loss' in k else f"{v:.2e}" if isinstance(v, float) and k == 'lr' else str(v) for k, v in result.items()]
                    results_table.add_row(*row_values, style="bold green" if result["epoch"] == best_loss_epoch else "")
                layout["table"].update(Panel(results_table, title="[blue]Training Log (Recent Epochs)[/blue]", border_style="blue"))
                
                # Step the learning rate scheduler
                if args.scheduler == 'plateau': scheduler.step(avg_total_loss)
                else: scheduler.step()

                # --- TensorBoard Logging ---
                writer.add_scalar('Loss/Adversarial', avg_adv_loss, epoch)
                if tv_weight > 0: writer.add_scalar('Loss/TotalVariation', avg_tv_loss, epoch)
                if nps_weight > 0: writer.add_scalar('Loss/NonPrintability', avg_nps_loss, epoch)
                if style_weight > 0: writer.add_scalar('Loss/Style', avg_style_loss, epoch)
                if pattern_weight > 0: 
                    writer.add_scalar('Loss/Pattern', avg_pattern_loss, epoch)
                    if target_camo_pattern is not None: writer.add_image('Target Camouflage Pattern', target_camo_pattern, epoch)
                writer.add_scalar('Loss/Total', avg_total_loss, epoch)
                writer.add_scalar('Learning_Rate', current_lr, epoch)
                writer.add_image('Adversarial Patch', adversarial_patch, epoch)
                torch.save({'epoch': epoch + 1, 'patch_state_dict': adversarial_patch.data.clone(), 'optimizer_state_dict': optimizer.state_dict(), 'scaler_state_dict': scaler.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'scheduler_type': args.scheduler}, os.path.join(log_dir, "patch_checkpoint.pth"))

                # Check for early stopping
                if not args.no_patience and epochs_no_improve >= early_stopping_patience:
                    stop_reason = f"Early stopping triggered after {early_stopping_patience} epochs"
                    live.stop(); break
    finally:
        # --- Cleanup and Final Summary ---
        if live.is_started: live.stop()
        console.show_cursor(True); writer.close()
        summary_message = (f"• Stop Reason: {stop_reason}\n"
                           f"• Total Epochs Trained: {final_epoch + 1}\n"
                           f"• Best Loss: {best_loss:.4f} (Epoch {best_loss_epoch})\n"
                           f"• Log Dir: {log_dir}")
        summary_panel = Panel(summary_message, title="[bold blue]Training Summary[/bold blue]", border_style="blue")
        console.print(summary_panel)
        send_notification("✅ Training Run Finished", summary_message)

# --- Utility Functions ---
def generate_run_name(parent_dir, num_patches_total, current_patch_num, args):
    """Generates a descriptive, unique directory name for the training run."""
    model_name_part = f"{len(args.models_to_target)}models"
    mode_part = f"{args.attack_mode}_{args.training_mode}"
    base_name = f"{datetime.now().strftime('%Y%m%d')}_{model_name_part}_{mode_part}_p{args.patch_size}"
    if num_patches_total > 1: base_name += f"_run{current_patch_num}"
    version = 1
    while True:
        run_name = f"{base_name}_v{version}"
        log_dir = os.path.join(parent_dir, run_name)
        if not os.path.exists(log_dir): return log_dir
        version += 1

def estimate_dataset_ram_usage(dataset_path, transform, num_samples=100):
    """
    Estimates the RAM required to pre-load the entire dataset by sampling a subset
    of images and extrapolating their memory usage.
    """
    console.print(f"🧠 [cyan]Estimating dataset RAM usage by sampling {num_samples} random images...[/cyan]")
    sample_dataset = VisDroneDatasetLazy(root_dir=dataset_path, transform=transform)
    total_images = len(sample_dataset)
    if total_images == 0: return 0
    
    num_samples = min(total_images, num_samples)
    indices_to_sample = random.sample(range(total_images), num_samples)
    
    total_bytes_for_samples = 0
    for i in indices_to_sample:
        sample = sample_dataset[i]
        if sample is not None and sample[0] is not None:
            total_bytes_for_samples += sample[0].nbytes
    
    if not len(indices_to_sample): return 0.1
    
    avg_bytes_per_item = total_bytes_for_samples / len(indices_to_sample)
    estimated_total_gb = (avg_bytes_per_item * total_images) / (1024**3)
    
    return estimated_total_gb

# --- Main Execution Block ---
def main(args):
    # Validate command-line arguments
    if args.attack_mode == 'misclassify' and args.decoy_class is None:
        console.print(f"💥 [bold red]Error: --decoy_class is required for 'misclassify' attack mode.[/bold red]"); sys.exit(1)
        
    # Load the config and select the settings for the specified training mode
    full_config = load_config(args.config)
    if args.training_mode not in full_config.get('training_modes', {}):
        console.print(f"💥 [bold red]Error: Training mode '{args.training_mode}' not found in config file.[/bold red]"); sys.exit(1)
    active_config = full_config['training_modes'][args.training_mode]
    
    # Populate args with top-level settings from the config for easier access
    args.models_to_target = full_config['models_to_target']
    args.dataset_path = full_config['dataset_path']
    args.target_classes = full_config['target_classes']
    args.patch_size = full_config['patch_size']

    # Setup GPU environment
    if args.gpu_ids:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpu_ids))
        console.print(f"✅ [green]Running on specified GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}[/green]")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu': console.print(f"⚠️ [yellow]CUDA not available. Running on CPU. This will be very slow.[/yellow]")

    # Setup logging and TensorBoard
    parent_dir = "runs"
    os.makedirs(parent_dir, exist_ok=True)
    tb = program.TensorBoard(); tb.configure(argv=[None, '--logdir', parent_dir]); url = tb.launch()
    console.print(Panel(f"🔌 [bold]TensorBoard is running: [link={url}]{url}[/link][/bold]", title="TensorBoard", border_style="blue"))

    # Validate incompatible arguments
    if args.starter_image and args.resume: console.print(f"[red]Error: --starter_image and --resume cannot be used together.[/red]"); sys.exit(1)
    if args.patches > 1 and args.resume: console.print("[bold yellow]Warning: --resume is ignored when generating multiple patches.[/bold yellow]"); args.resume = None

    # Load and compile models
    models = []
    for model_name in args.models_to_target:
        console.print(f"⏳ [magenta]Loading model: {model_name}...[/magenta]")
        model = YOLO(model_name).to(device)
        model.model.eval()
        # Removed channels_last conversion for model to avoid potential compile/kernel issues
        if device.type == 'cuda' and not args.no_compile:
            try:
                model.model = torch.compile(model.model)
                console.print(f"✅ [green]Model '{model_name}' compiled successfully.[/green]")
            except Exception as e:
                console.print(f"⚠️ [yellow]torch.compile() failed for '{model_name}': {e}. Running without compilation.[/yellow]")
        models.append(model)

    if not os.path.exists(args.dataset_path): console.print(f"[red]Error: Dataset path not found: '{args.dataset_path}'[/red]"); sys.exit(1)
    
    # Determine dataset loading strategy based on available RAM
    transform = T.Compose([T.Resize((640, 640)), T.ToTensor()])
    required_gb = estimate_dataset_ram_usage(args.dataset_path, transform) * 1.2
    available_gb = psutil.virtual_memory().available / (1024**3)
    console.print(f"🧠 [cyan]Memory Check: Available RAM: {available_gb:.2f} GB, Estimated required: {required_gb:.2f} GB[/cyan]")

    if available_gb > required_gb:
        console.print(f"🚀 [magenta]Sufficient RAM detected. Using high-performance RAM pre-loading strategy.[/magenta]")
        dataset = VisDroneDatasetPreload(root_dir=args.dataset_path, transform=transform)
        num_workers = 0 
    else:
        console.print(f"💾 [magenta]Insufficient RAM for pre-loading. Using memory-safe on-demand disk loading.[/magenta]")
        dataset = VisDroneDatasetLazy(root_dir=args.dataset_path, transform=transform)
        num_workers = min(os.cpu_count(), 16)

    hp = active_config['hyperparameters']
    
    # Determine batch size (manual, resume, or autotune)
    if args.batch_size:
        final_batch_size = args.batch_size
        console.print(f"🔩 [bold yellow]Manual batch size set to {final_batch_size}. Autotuning bypassed.[/bold yellow]")
    elif args.resume:
        final_batch_size = torch.load(args.resume).get('batch_size', hp['base_batch_size'])
        console.print(f"🔄 Resuming with batch size {final_batch_size} from checkpoint.")
    elif device.type == 'cuda':
        model_to_tune = models[0] 
        final_batch_size = find_optimal_batch_size(model_to_tune, device, len(dataset), initial_batch_size=16)
    else:
        final_batch_size = hp['base_batch_size']
        console.print(f"⚙️ Using default batch size from config: {final_batch_size}")
        
    hp['base_batch_size'] = final_batch_size
    
    # Create the DataLoader
    pin_memory = (device.type == 'cuda')
    if num_workers and num_workers > 0:
        dataloader = DataLoader(
            dataset,
            batch_size=final_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            persistent_workers=True,
            prefetch_factor=4,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=final_batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            persistent_workers=False,
        )

    # Register the signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)

    # Loop to generate multiple patches if requested
    for i in range(args.patches):
        console.print("\n" + "="*80, style="bold blue")
        console.print(f"🚀🚀 [bold magenta] STARTING PATCH GENERATION RUN {i + 1} of {args.patches} [/bold magenta] 🚀🚀🚀")
        console.print("="*80, style="bold blue")
        
        try:
            log_dir = generate_run_name(parent_dir, args.patches, i + 1, args)
            os.makedirs(log_dir, exist_ok=True)
            console.print(f"📝 [cyan]Logging run to: {log_dir}[/cyan]")
            
            # Save the exact configuration used for this run for reproducibility
            run_params = {
                'command_line_args': vars(args),
                'active_config': active_config
            }
            with open(os.path.join(log_dir, 'run_parameters.json'), 'w') as f:
                json.dump(run_params, f, indent=4)
            console.print(f"✅ [green]Run parameters saved to {os.path.join(log_dir, 'run_parameters.json')}[/green]")

            # Start the training process
            train_adversarial_patch(
                config=active_config, models=models, log_dir=log_dir, device=device,
                dataloader=dataloader, num_workers=num_workers, pin_memory=pin_memory,
                args=args, resume_path=args.resume, starter_image_path=args.starter_image
            )
        except Exception as e:
            # Catch exceptions to allow subsequent runs to continue
            console.print("\n" + "="*60, f"💥 [bold red]An unexpected error occurred during run {i+1}! Moving to next run...[/bold red]", "="*60, sep="\n")
            error_info = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            console.print(error_info)
            send_notification("❌ Training Run Crashed", f"Run {i+1} of {args.patches} crashed.\n\n{error_info}", tags="rotating_light")
            continue
    
    console.print(f"\n✅ [bold green]All {args.patches} patch generation runs are complete.[/bold green]")

if __name__ == '__main__':
    # Set multiprocessing strategy for compatibility
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    # --- Command-Line Argument Parser ---
    parser = argparse.ArgumentParser(description="Train adversarial patches against YOLO models (v1.9).")
    parser.add_argument('--config', type=str, default='config.json', help='Path to the JSON configuration file.')
    parser.add_argument('--training_mode', type=str, default='normal', choices=['normal', 'covert_style', 'covert_procedural'], help='The training methodology to use.')
    parser.add_argument('--attack_mode', type=str, default='hide', choices=['hide', 'misclassify'], help='The adversarial attack strategy.')
    parser.add_argument('--decoy_class', type=int, default=None, help='The target class ID for misclassification attacks.')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume a specific run.')
    parser.add_argument('--starter_image', type=str, default=None, help='Path to an image to use as the starting point for the patch.')
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=None, help='Specific GPU IDs to use (e.g., 0 1 2).')
    parser.add_argument('--patches', type=int, default=1, help='Number of patches to generate by running the script multiple times.')
    parser.add_argument('--scheduler', type=str, default='cosine_warm', choices=['plateau', 'cosine_warm'], help='Learning rate scheduler to use.')
    parser.add_argument('--no-compile', action='store_true', help='Disable torch.compile() for the model.')
    parser.add_argument('--no-patience', action='store_true', help='Disable early stopping based on patience.')
    parser.add_argument('--patch_coverage', type=float, default=0.35, help='The desired patch coverage of the target object\'s area (default: 0.35 for 35%%).')
    parser.add_argument('--augmentations', action='store_true', help='Enable a set of augmentations to simulate real-world conditions.')
    parser.add_argument('--batch_size', type=int, default=None, help='Manually set the batch size and bypass autotuning.')

    args = parser.parse_args()
    
    try:
        main(args)
    except SystemExit:
        pass
    except Exception as e:
        console.print(f"💥 [bold red]A critical error occurred in the main script execution.[/bold red]")
        error_info = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        console.print(error_info)
        send_notification("❌ Critical Script Failure", error_info, tags="rotating_light")
    finally:
        console.show_cursor(True)
        sys.exit(0)
