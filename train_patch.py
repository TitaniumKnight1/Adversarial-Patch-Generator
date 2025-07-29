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
try:
    import noise
except ImportError:
    print("The 'noise' library is not installed. Please install it using: pip install noise")
    sys.exit(1)
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts
from datetime import datetime
import math
import signal
import torch.multiprocessing
import psutil
import json
import requests

# --- Rich CLI Components for a clean UI ---
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.layout import Layout
from rich.panel import Panel

# --- Initialize Rich Console ---
console = Console()

def send_notification(title, message, tags="tada"):
    """Sends a notification to a ntfy.sh topic for remote monitoring."""
    try:
        requests.post(
            "https://ntfy.sh/PatchTraining",  # Replace with your own ntfy.sh topic if desired
            data=message.encode(encoding='utf-8'),
            headers={
                "Title": title.encode(encoding='utf-8'),
                "Tags": tags
            }
        )
    except Exception as e:
        console.print(f"⚠️ [yellow]Failed to send notification: {e}[/yellow]")

def load_config(config_path):
    """Loads the training configuration from a JSON file."""
    if not os.path.exists(config_path):
        console.print(f"💥 [bold red]Config file not found at '{config_path}'. Please create it.[/bold red]")
        sys.exit(1)
    with open(config_path, 'r') as f:
        config = json.load(f)
    console.print(f"✅ [green]Configuration loaded from '{config_path}'[/green]")
    return config

def collate_fn(batch):
    """Custom collate function to filter out None values from a batch, which can occur if an image fails to load."""
    batch = list(filter(lambda x: x is not None and x[0] is not None, batch))
    if not batch:
        return None, None, None, None
    images, box_label_pairs, patch_locations = zip(*batch)
    boxes, labels = zip(*box_label_pairs)
    return torch.stack(images, 0), boxes, labels, torch.stack(patch_locations, 0)

# --- Dataset Classes ---

class VisDroneDatasetLazy(Dataset):
    """
    Dataset class for VisDrone. Loads images and annotations from disk on-the-fly.
    This is memory-efficient for large datasets that cannot fit into RAM.
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
            # Return None if an image is corrupted or cannot be opened
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
        
        # Scale bounding boxes to the transformed image size (640x640)
        if boxes_tensor.nelement() > 0:
            scale_x, scale_y = 640 / original_size[0], 640 / original_size[1]
            boxes_tensor[:, [0, 2]] *= scale_x
            boxes_tensor[:, [1, 3]] *= scale_y
        
        patch_location = torch.zeros(4) # Placeholder for patch location
        return image, (boxes_tensor, labels_tensor), patch_location

class VisDroneDatasetPreload(Dataset):
    """
    Dataset class for VisDrone that pre-loads and pre-processes the entire dataset into RAM.
    This offers significant speed improvements during training if sufficient RAM is available.
    It also caches the processed dataset to disk for faster subsequent runs.
    """
    def __init__(self, root_dir, transform=None, cache_file="preprocessed_dataset.pth"):
        cache_path = os.path.join(root_dir, cache_file)
        if os.path.exists(cache_path):
            console.print(f"⏳ [blue]Loading pre-processed dataset from cache: {cache_path}[/blue]")
            try:
                # Load the cached data
                cached_data = torch.load(cache_path, weights_only=False)
                # Basic check to ensure cache format is as expected
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
    """A dummy dataset that generates random data, used for batch size autotuning without loading real data."""
    def __init__(self, length=2048, image_size=(3, 640, 640)):
        self.length, self.image_size = length, image_size
    def __len__(self): return self.length
    def __getitem__(self, idx): return torch.rand(self.image_size)

# --- Loss Functions and Patch Regularizers ---

class TotalVariationLoss(torch.nn.Module):
    """Computes the total variation loss, which encourages spatial smoothness in the generated patch."""
    def forward(self, patch):
        if patch.dim() == 3: patch = patch.unsqueeze(0)
        wh_diff = torch.sum(torch.abs(patch[:, :, :, 1:] - patch[:, :, :, :-1]))
        ww_diff = torch.sum(torch.abs(patch[:, :, 1:, :] - patch[:, :, :-1, :]))
        return (wh_diff + ww_diff) / (patch.size(2) * patch.size(3))

class ColorDiversityLoss(torch.nn.Module):
    """
    Computes a loss that encourages color diversity in the patch.
    The loss is the negative mean standard deviation of the color channels.
    Minimizing this loss maximizes the standard deviation, thus encouraging more varied colors.
    """
    def forward(self, patch):
        if patch.dim() == 3:
            patch = patch.unsqueeze(0)
        # Flatten the spatial dimensions
        patch_flat = patch.view(patch.size(0), patch.size(1), -1)
        # Return the negative mean standard deviation across color channels
        return -torch.mean(torch.std(patch_flat, dim=2))

def generate_camouflage_pattern(width, height, colors, scale=25.0, octaves=4, persistence=0.6, lacunarity=2.0, device='cpu'):
    """
    Generates a procedural camouflage pattern using Perlin noise.
    This creates a natural-looking texture based on a provided color palette.
    """
    num_colors = colors.shape[0]
    world = np.zeros((height, width))
    
    # Generate Perlin noise field
    for i in range(height):
        for j in range(width):
            world[i][j] = noise.pnoise2(i / scale, 
                                        j / scale, 
                                        octaves=octaves, 
                                        persistence=persistence, 
                                        lacunarity=lacunarity, 
                                        repeatx=width, 
                                        repeaty=height, 
                                        base=random.randint(0, 100))

    # Normalize noise to map to color indices
    world_normalized = (world - np.min(world)) / (np.max(world) - np.min(world))
    color_indices = np.floor(world_normalized * num_colors).astype(np.int32)
    color_indices[color_indices == num_colors] = num_colors - 1

    # Create the pattern and apply a slight blur for smoothness
    camo_pattern_np = colors[color_indices].cpu().numpy()
    camo_pattern_tensor = torch.from_numpy(camo_pattern_np).permute(2, 0, 1).to(device)
    blurred_pattern = TF.gaussian_blur(camo_pattern_tensor.unsqueeze(0), kernel_size=[3, 3]).squeeze(0)
    
    return blurred_pattern

# --- Autotuning and Training Logic ---

def autotune_batch_size(device, models, dataset_len, initial_batch_size=2, safety_factor=0.9):
    """
    Automatically finds the maximum batch size that fits into VRAM.
    It performs a two-phase search: a rapid exponential probe followed by a binary search refinement.
    """
    model_to_tune = models[0] # Use the first model for tuning
    
    with console.status("[bold yellow]Starting batch size autotune...") as status:
        batch_size = initial_batch_size
        last_success_size, oom_size = 1, -1

        # Phase 1: Exponential probing to quickly find the OOM boundary
        while True:
            if batch_size > dataset_len:
                status.update(f"[bold yellow]Probe exceeds dataset size. Max size is {last_success_size}.[/bold yellow]")
                oom_size = batch_size
                break
            
            status.update(f"[bold yellow]Phase 1/2 (Probing): Testing batch size {batch_size}...[/bold yellow]")
            try:
                dummy_data = DummyDataset(length=batch_size)
                dataloader = DataLoader(dummy_data, batch_size=batch_size)
                images = next(iter(dataloader)).to(device)
                images.requires_grad_(True)
                with torch.amp.autocast(device.type):
                    output = model_to_tune.model(images)
                    dummy_loss = output[0].sum()
                dummy_loss.backward()
                
                last_success_size = batch_size
                batch_size *= 2
                del images, dataloader, output, dummy_loss
                torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                oom_size = batch_size
                torch.cuda.empty_cache()
                break
            except StopIteration:
                last_success_size = batch_size
                oom_size = batch_size * 2
                break

        # Phase 2: Binary search to refine the exact maximum batch size
        low, high, best_size = last_success_size, oom_size - 1, last_success_size
        while low <= high:
            mid = (low + high) // 2
            if mid <= best_size:
                low = mid + 1
                continue

            status.update(f"[bold yellow]Phase 2/2 (Refining): Testing batch size {mid}...[/bold yellow]")
            try:
                dummy_data = DummyDataset(length=mid)
                dataloader = DataLoader(dummy_data, batch_size=mid)
                images = next(iter(dataloader)).to(device)
                images.requires_grad_(True)
                with torch.amp.autocast(device.type):
                    output = model_to_tune.model(images)
                    dummy_loss = output[0].sum()
                dummy_loss.backward()

                best_size = mid
                low = mid + 1
                del images, dataloader, output, dummy_loss
                torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                high = mid - 1
                torch.cuda.empty_cache()
            except StopIteration:
                best_size = mid
                break
    
    final_size = int(best_size * safety_factor)
    console.print(f"✅ [green]Autotune complete. Max batch size: {best_size}. Applying {int((1-safety_factor)*100)}% safety margin -> {final_size}[/green]")
    return max(1, final_size)

# Global state for signal handler to allow graceful exit
training_state = {'best_patch': None, 'log_dir': None, 'should_exit': False}

def signal_handler(sig, frame):
    """Handles Ctrl+C interrupts to save the best patch before exiting."""
    console.print("\n[bold yellow]Ctrl+C detected. Saving best patch and exiting gracefully...[/bold yellow]")
    if training_state['best_patch'] is not None and training_state['log_dir'] is not None:
        save_path = os.path.join(training_state['log_dir'], "best_patch_interrupted.png")
        try:
            T.ToPILImage()(training_state['best_patch'].cpu()).save(save_path)
            console.print(f"✅ [green]Best patch saved to: {save_path}[/green]")
        except Exception as e:
            console.print(f"💥 [red]Could not save best patch: {e}[/red]")
    training_state['should_exit'] = True

def train_adversarial_patch(config, models, log_dir, device, dataloader, args, resume_path=None, starter_image_path=None):
    """The main training loop for the adversarial patch."""
    if device.type == 'cuda': cudnn.benchmark = True
    writer = SummaryWriter(log_dir=log_dir)
    
    hp, lw = config['hyperparameters'], config['loss_weights']
    patch_size, learning_rate = config['patch_size'], hp['base_learning_rate']
    
    # Loss weights
    adv_weight = lw.get('adv_weight', 100.0)
    pattern_weight = lw.get('pattern_weight', 5.0)
    tv_weight = lw.get('tv_weight', 1.0)
    color_weight = lw.get('color_weight', 2.0)

    max_epochs, target_classes = hp['max_epochs'], config.get('target_classes', [])

    # Initialize the patch either from a starter image or random noise
    if starter_image_path and os.path.exists(starter_image_path):
        console.print(f"🌱 [cyan]Initializing patch from starter image: {starter_image_path}[/cyan]")
        starter_image = Image.open(starter_image_path).convert("RGB")
        transform_starter = T.Compose([T.Resize((patch_size, patch_size)), T.ToTensor()])
        adversarial_patch = transform_starter(starter_image).to(device, non_blocking=True)
        adversarial_patch.requires_grad_(True)
    else:
        console.print(f"🎨 [cyan]Initializing patch with random noise.[/cyan]")
        adversarial_patch = torch.rand((3, patch_size, patch_size), device=device, requires_grad=True)

    optimizer = torch.optim.AdamW([adversarial_patch], lr=learning_rate, amsgrad=True)
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))
    
    # Configure the learning rate scheduler
    if args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=hp['plateau_patience'])
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
    elif args.scheduler == 'cosine_warm':
        restart_epochs = hp.get('cosine_restart_epochs', max(1, max_epochs // 4))
        console.print(f"🔥 [bold cyan]Using CosineAnnealingWarmRestarts with T_0 = {restart_epochs} epochs.[/bold cyan]")
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=restart_epochs, T_mult=1, eta_min=1e-6)
    else: # Default
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=hp['plateau_patience'])

    # Initialize loss functions
    total_variation = TotalVariationLoss().to(device)
    color_diversity = ColorDiversityLoss().to(device)
    pattern_loss_fn = torch.nn.L1Loss().to(device) if args.covert else None

    # Resume training from a checkpoint if provided
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

    # --- Rich Layout for Training UI ---
    model_names_str = ", ".join([os.path.basename(m.ckpt_path) for m in models])
    target_classes_str = "All" if not target_classes else str(target_classes)
    layout = Layout()
    
    config_panel_content = (f"🚀 [bold magenta]Starting Adversarial Patch Training[/bold magenta]\n"
                          f"   - [b]Target Models[/b]: [cyan]{model_names_str}[/cyan]\n"
                          f"   - [b]Target Classes[/b]: [cyan]{target_classes_str}[/cyan]\n"
                          f"   - [b]Device[/b]: [cyan]{device.type.upper()}[/cyan]\n"
                          f"   - [b]Batch Size[/b]: [cyan]{dataloader.batch_size}[/cyan]\n"
                          f"   - [b]Patch Coverage[/b]: [cyan]{args.patch_coverage:.0%}[/cyan]\n"
                          f"   - [b]Scheduler[/b]: [cyan]{args.scheduler}[/cyan]\n"
                          f"   - [b]Learning Rate[/b]: [cyan]{learning_rate:.2e}[/cyan]\n"
                          f"   - [b]Loss Weights[/b]: Adv (Hide): {adv_weight:.1e}")
    
    config_panel_content += f", TV: {tv_weight:.1e}" if not args.no_tv_loss else ", TV: [red]Disabled[/red]"
    config_panel_content += f", Pattern: {pattern_weight:.1e}" if args.covert else ""
    config_panel_content += f", Color: {color_weight:.1e}" if not args.no_color_loss else ", Color: [red]Disabled[/red]"

    layout.split_column(
        Layout(Panel(config_panel_content, title="[yellow]Training Configuration[/yellow]", border_style="yellow"), name="config"),
        Layout(name="progress", size=3),
        Layout(name="table", ratio=1)
    )
    progress = Progress(TextColumn("[bold blue]{task.description}", justify="right"), BarColumn(bar_width=None), "[progress.percentage]{task.percentage:>3.1f}%", "•", MofNCompleteColumn(), "•", TimeRemainingColumn(), console=console)
    layout["progress"].update(Panel(progress, title="[yellow]Current Epoch Progress[/yellow]", border_style="yellow"))

    best_loss = float('inf')
    epochs_no_improve, best_loss_epoch = 0, -1
    epoch_results = []
    stop_reason = "Max epochs reached"
    final_epoch = 0
    
    live = Live(layout, screen=True, redirect_stderr=False, vertical_overflow="visible", console=console)
    
    try:
        with live:
            for epoch in range(start_epoch, hp['max_epochs']):
                if training_state['should_exit']:
                    stop_reason = "Manually interrupted (Ctrl+C)"
                    break
                
                final_epoch = epoch
                epoch_start_time = time.time()
                total_adv_loss, total_tv_loss, total_pattern_loss, total_color_loss = 0, 0, 0, 0
                task_id = progress.add_task(f"Epoch {epoch + 1}", total=len(dataloader))

                # --- Generate a new camouflage pattern for each epoch in Covert Mode ---
                target_camo_pattern = None
                if args.covert:
                    try:
                        first_images = next(iter(dataloader))[0].to(device)
                        # Extract a color palette from the first image of the batch
                        pixels = first_images[0].permute(1, 2, 0).reshape(-1, 3)
                        for _ in range(5): # Try a few times to get a diverse palette
                            palette_pool = pixels[torch.randperm(pixels.shape[0])[:1000]]
                            color_palette = palette_pool[torch.randperm(palette_pool.shape[0])[:4]]
                            if torch.std(color_palette) > 0.1: break
                        target_camo_pattern = generate_camouflage_pattern(patch_size, patch_size, color_palette, device=device)
                    except StopIteration:
                        console.print("[yellow]Dataloader empty, cannot generate camo pattern.[/yellow]")

                for i, batch_data in enumerate(dataloader):
                    if training_state['should_exit']: break
                    if batch_data is None or batch_data[0] is None:
                        progress.update(task_id, advance=1); continue
                    
                    images, gt_boxes_batch, gt_labels_batch, _ = batch_data
                    images = images.to(device, non_blocking=True)
                    
                    adversarial_patch.data.clamp_(0,1)
                    
                    # --- STAGE 1: Adversarial Optimization (Hiding Attack) ---
                    optimizer.zero_grad(set_to_none=True)
                    
                    patched_images_adv = images.clone()
                    for img_idx in range(images.size(0)):
                        gt_boxes, gt_labels = gt_boxes_batch[img_idx], gt_labels_batch[img_idx]
                        targetable_indices = [j for j, label in enumerate(gt_labels) if label.item() in target_classes] if target_classes else list(range(len(gt_boxes)))

                        if not targetable_indices: continue
                        
                        # Apply patch to a random target object in the image
                        box = gt_boxes[random.choice(targetable_indices)].int()
                        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                        
                        target_w, target_h = x2 - x1, y2 - y1
                        if target_w <=0 or target_h <=0: continue
                        
                        # Calculate patch size based on coverage and apply it
                        patch_area = target_w * target_h * args.patch_coverage
                        patch_size_to_apply = max(20, min(int(math.sqrt(patch_area)), min(images.shape[2], images.shape[3])))
                        resized_patch = TF.resize(adversarial_patch.unsqueeze(0), (patch_size_to_apply, patch_size_to_apply), antialias=True).squeeze(0)
                        
                        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                        patch_x = max(0, min(center_x - patch_size_to_apply // 2, images.shape[3] - patch_size_to_apply))
                        patch_y = max(0, min(center_y - patch_size_to_apply // 2, images.shape[2] - patch_size_to_apply))
                        
                        patched_images_adv[img_idx, :, patch_y:patch_y+patch_size_to_apply, patch_x:patch_x+patch_size_to_apply] = resized_patch

                    # Forward pass through a randomly selected model from the ensemble
                    with torch.amp.autocast(device.type):
                        raw_preds = random.choice(models).model(patched_images_adv)[0].transpose(1, 2)
                        objectness_scores = raw_preds[..., 4]
                        
                        # The goal is to maximize objectness, which corresponds to minimizing confidence of detection
                        adversarial_loss = objectness_scores.mean()
                        tv_loss = total_variation(adversarial_patch) if not args.no_tv_loss else torch.tensor(0.0, device=device)
                        color_div_loss = color_diversity(adversarial_patch) if not args.no_color_loss else torch.tensor(0.0, device=device)
                        
                        adv_step_loss = (adv_weight * adversarial_loss) + (tv_weight * tv_loss) + (color_weight * color_div_loss)

                    scaler.scale(adv_step_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    adversarial_patch.data.clamp_(0, 1)

                    # --- STAGE 2: Camouflage Pattern Matching (Covert Mode Only) ---
                    pattern_loss = torch.tensor(0.0, device=device)
                    if args.covert and pattern_loss_fn is not None and target_camo_pattern is not None:
                        optimizer.zero_grad(set_to_none=True)
                        with torch.amp.autocast(device.type):
                            pattern_loss = pattern_loss_fn(adversarial_patch, target_camo_pattern)
                            camo_step_loss = pattern_weight * pattern_loss
                        scaler.scale(camo_step_loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        adversarial_patch.data.clamp_(0, 1)

                    total_adv_loss += adversarial_loss.item()
                    total_tv_loss += tv_loss.item()
                    total_pattern_loss += pattern_loss.item()
                    total_color_loss += color_div_loss.item()
                    
                    progress.update(task_id, advance=1)

                progress.remove_task(task_id)
                avg_adv_loss = total_adv_loss / len(dataloader)
                avg_tv_loss = total_tv_loss / len(dataloader)
                avg_pattern_loss = total_pattern_loss / len(dataloader)
                avg_color_loss = total_color_loss / len(dataloader)
                
                avg_total_loss = (adv_weight * avg_adv_loss) + (tv_weight * avg_tv_loss) + \
                                 (pattern_weight * avg_pattern_loss) + (color_weight * avg_color_loss)

                # --- Logging and Checkpointing ---
                if avg_total_loss < best_loss:
                    best_loss, epochs_no_improve, best_loss_epoch = avg_total_loss, 0, epoch + 1
                    training_state['best_patch'] = adversarial_patch.data.clone()
                    training_state['log_dir'] = log_dir
                    torch.save({'epoch': epoch + 1, 'patch_state_dict': adversarial_patch.data.clone(), 'optimizer_state_dict': optimizer.state_dict(), 'scaler_state_dict': scaler.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'scheduler_type': args.scheduler}, os.path.join(log_dir, "best_patch_checkpoint.pth"))
                    T.ToPILImage()(adversarial_patch.cpu()).save(os.path.join(log_dir, "best_patch.png"))
                else:
                    epochs_no_improve += 1
                
                # Update Rich table with epoch results
                patience_str = f"{epochs_no_improve}" if args.no_patience else f"{epochs_no_improve}/{hp['early_stopping_patience']}"
                result_row = {"epoch": epoch + 1, "duration": f"{time.time() - epoch_start_time:.1f}", "adv_loss": avg_adv_loss, "tv_loss": avg_tv_loss, "pattern_loss": avg_pattern_loss, "color_loss": avg_color_loss, "total_loss": avg_total_loss, "lr": optimizer.param_groups[0]['lr'], "patience": patience_str}
                epoch_results.append(result_row)
                
                results_table = Table(title="Epoch Results", expand=True, border_style="blue")
                for col in result_row.keys(): results_table.add_column(col)
                
                for result in epoch_results[-20:]: # Display last 20 epochs
                    row_values = [f"{v:.4e}" if isinstance(v, float) and 'loss' in k else f"{v:.2e}" if isinstance(v, float) and k == 'lr' else str(v) for k, v in result.items()]
                    results_table.add_row(*row_values, style="bold green" if result["epoch"] == best_loss_epoch else "")
                layout["table"].update(Panel(results_table, title="[blue]Training Log (Recent Epochs)[/blue]", border_style="blue"))
                
                # Step the scheduler
                if args.scheduler == 'plateau':
                    scheduler.step(avg_total_loss)
                else:
                    scheduler.step()

                # TensorBoard logging
                writer.add_scalar('Loss/Adversarial', avg_adv_loss, epoch)
                writer.add_scalar('Loss/TotalVariation', avg_tv_loss, epoch)
                writer.add_scalar('Loss/ColorDiversity', avg_color_loss, epoch)
                writer.add_scalar('Loss/Pattern', avg_pattern_loss, epoch)
                writer.add_scalar('Loss/Total', avg_total_loss, epoch)
                writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
                writer.add_image('Adversarial Patch', adversarial_patch, epoch)
                if args.covert and target_camo_pattern is not None:
                    writer.add_image('Target Camouflage Pattern', target_camo_pattern, epoch)
                
                # Save latest checkpoint
                torch.save({'epoch': epoch + 1, 'patch_state_dict': adversarial_patch.data.clone(), 'optimizer_state_dict': optimizer.state_dict(), 'scaler_state_dict': scaler.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'scheduler_type': args.scheduler}, os.path.join(log_dir, "patch_checkpoint.pth"))

                if not args.no_patience and epochs_no_improve >= hp['early_stopping_patience']:
                    stop_reason = f"Early stopping triggered after {hp['early_stopping_patience']} epochs"
                    break
        
    finally:
        if live.is_started: live.stop()
        console.show_cursor(True)
        writer.close()
        summary_message = (f"• Stop Reason: {stop_reason}\n"
                           f"• Total Epochs Trained: {final_epoch + 1}\n"
                           f"• Best Loss: {best_loss:.4f} (Epoch {best_loss_epoch})\n"
                           f"• Log Dir: {log_dir}")
        console.print(Panel(summary_message, title="[bold blue]Training Summary[/bold blue]", border_style="blue"))
        send_notification("✅ Training Run Finished", summary_message)

def generate_run_name(config, parent_dir, num_patches_total, current_patch_num, args):
    """Generates a descriptive and unique directory name for the training run."""
    models_config = config['models_to_target']
    model_name_part = f"{len(models_config)}models" if isinstance(models_config, list) else os.path.splitext(os.path.basename(models_config))[0]
    mode_part = "covert" if args.covert else "normal"
    base_name = f"{datetime.now().strftime('%Y%m%d')}_{model_name_part}_{mode_part}_p{config['patch_size']}"
    if num_patches_total > 1: base_name += f"_run{current_patch_num}"
    
    version = 1
    while True:
        run_name = f"{base_name}_v{version}"
        log_dir = os.path.join(parent_dir, run_name)
        if not os.path.exists(log_dir): return log_dir
        version += 1

def estimate_dataset_ram_usage(dataset_path, transform, num_samples=100):
    """Estimates the RAM required to pre-load the entire dataset by sampling a subset of images."""
    console.print(f"🧠 [cyan]Estimating dataset RAM usage by sampling {num_samples} random images...[/cyan]")
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss
    sample_dataset = VisDroneDatasetLazy(root_dir=dataset_path, transform=transform)
    total_images = len(sample_dataset)
    if total_images == 0: return 0
    
    num_samples = min(total_images, num_samples)
    indices_to_sample = random.sample(range(total_images), num_samples)
    samples = [sample for i in indices_to_sample if (sample := sample_dataset[i]) is not None and sample[0] is not None]
    
    mem_after = process.memory_info().rss
    mem_used_by_samples = mem_after - mem_before
    del samples
    if not indices_to_sample: return 0.1
    
    avg_bytes_per_item = mem_used_by_samples / len(indices_to_sample)
    return (avg_bytes_per_item * total_images) / (1024**3) # Return in GB

def main(args):
    config = load_config('config.json')
    
    if args.gpu_ids:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpu_ids))
        console.print(f"✅ [green]Running on specified GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}[/green]")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu': console.print(f"⚠️ [yellow]CUDA not available. Running on CPU. This will be very slow.[/yellow]")

    parent_dir = "runs"
    os.makedirs(parent_dir, exist_ok=True)
    tb = program.TensorBoard(); tb.configure(argv=[None, '--logdir', parent_dir]); url = tb.launch()
    console.print(Panel(f"🔌 [bold]TensorBoard is running: [link={url}]{url}[/link][/bold]", title="TensorBoard", border_style="blue"))

    if args.starter_image and args.resume: console.print(f"[red]Error: --starter_image and --resume cannot be used together.[/red]"); sys.exit(1)
    if args.patches > 1 and args.resume: console.print("[bold yellow]Warning: --resume is ignored when generating multiple patches.[/bold yellow]"); args.resume = None

    # Load and compile models
    models = []
    for model_name in config['models_to_target']:
        console.print(f"⏳ [magenta]Loading model: {model_name}...[/magenta]")
        model = YOLO(model_name).to(device)
        model.model.eval()
        if device.type == 'cuda' and not args.no_compile:
            try:
                model.model = torch.compile(model.model)
                console.print(f"✅ [green]Model '{model_name}' compiled successfully.[/green]")
            except Exception as e:
                console.print(f"⚠️ [yellow]torch.compile() failed for '{model_name}': {e}. Running without compilation.[/yellow]")
        models.append(model)

    dataset_path = config['dataset_path']
    if not os.path.exists(dataset_path): console.print(f"[red]Error: Dataset path not found: '{dataset_path}'[/red]"); sys.exit(1)
    
    # Determine batch size
    if args.batch_size:
        final_batch_size = args.batch_size
        console.print(f"✅ [bold yellow]Manual batch size specified: {final_batch_size}. Skipping auto-tuner.[/bold yellow]")
    elif device.type == 'cuda':
        num_images = len(os.listdir(os.path.join(dataset_path, 'images')))
        final_batch_size = autotune_batch_size(device, models, num_images, config['hyperparameters']['base_batch_size'])
    else:
        final_batch_size = config['hyperparameters']['base_batch_size']
        
    # Determine dataset loading strategy based on available RAM
    transform = T.Compose([T.Resize((640, 640)), T.ToTensor()])
    required_gb = estimate_dataset_ram_usage(dataset_path, transform) * 1.2 # 20% buffer
    available_gb = psutil.virtual_memory().available / (1024**3)
    console.print(f"🧠 [cyan]Memory Check: Available RAM: {available_gb:.2f} GB, Estimated required: {required_gb:.2f} GB[/cyan]")

    if available_gb > required_gb:
        console.print(f"🚀 [magenta]Sufficient RAM detected. Using high-performance RAM pre-loading strategy.[/magenta]")
        dataset = VisDroneDatasetPreload(root_dir=dataset_path, transform=transform)
        num_workers = 0 
    else:
        console.print(f"💾 [magenta]Insufficient RAM for pre-loading. Using memory-safe on-demand disk loading.[/magenta]")
        dataset = VisDroneDatasetLazy(root_dir=dataset_path, transform=transform)
        num_workers = min(os.cpu_count(), 16)
    
    dataloader = DataLoader(dataset, batch_size=final_batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device.type == 'cuda'), collate_fn=collate_fn, persistent_workers=num_workers > 0)

    signal.signal(signal.SIGINT, signal_handler)

    for i in range(args.patches):
        console.print(Panel(f"🚀 [bold magenta] STARTING PATCH GENERATION RUN {i + 1} of {args.patches} [/bold magenta] 🚀", style="bold blue"))
        try:
            log_dir = generate_run_name(config, parent_dir, args.patches, i + 1, args)
            os.makedirs(log_dir, exist_ok=True)
            console.print(f"📝 [cyan]Logging run to: {log_dir}[/cyan]")

            train_adversarial_patch(
                config=config, models=models, log_dir=log_dir, device=device,
                dataloader=dataloader, args=args, resume_path=args.resume, 
                starter_image_path=args.starter_image
            )
        except Exception as e:
            console.print(f"\n💥 [bold red]An unexpected error occurred during run {i+1}! Moving to next run...[/bold red]")
            error_info = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            console.print(error_info)
            send_notification("❌ Training Run Crashed", f"Run {i+1} of {args.patches} crashed.\n\n{error_info}", tags="rotating_light")
            continue
    
    console.print(f"\n✅ [bold green]All {args.patches} patch generation runs are complete.[/bold green]")

if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    parser = argparse.ArgumentParser(description="Train adversarial patches against YOLO models.")
    parser.add_argument('--batch_size', type=int, default=None, help="Manually specify the batch size, overriding the auto-tuner.")
    parser.add_argument('--resume', type=str, default=None, help='Path to a checkpoint to resume training from.')
    parser.add_argument('--starter_image', type=str, default=None, help='Path to an image to use as the starting point for the patch.')
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=None, help='Specific GPU IDs to use (e.g., 0 1 2).')
    parser.add_argument('--patches', type=int, default=1, help='Number of patches to generate by running the script multiple times.')
    parser.add_argument('--covert', action='store_true', help="Enable covert mode for camouflage-style patches.")
    parser.add_argument('--scheduler', type=str, default='cosine_warm', choices=['plateau', 'cosine', 'cosine_warm'], help='Learning rate scheduler to use.')
    parser.add_argument('--no-tv-loss', action='store_true', help='Disable the total variation loss constraint.')
    parser.add_argument('--no-color-loss', action='store_true', help='Disable the color diversity loss constraint.')
    parser.add_argument('--no-compile', action='store_true', help='Disable torch.compile() for the model.')
    parser.add_argument('--no-patience', action='store_true', help='Disable early stopping.')
    parser.add_argument('--patch_coverage', type=float, default=0.35, help="The desired patch coverage of the target object's area (default: 0.35 for 35%%).")

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
