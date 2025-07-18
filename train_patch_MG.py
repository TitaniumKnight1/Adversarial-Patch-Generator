# =================================================================================================
#           HIGH-PERFORMANCE DISTRIBUTED TRAINING SCRIPT (DDP)
# =================================================================================================
#
# HOW TO RUN THIS SCRIPT:
# -----------------------
# This script is designed to be launched with `torchrun` for optimal multi-GPU performance.
# `torchrun` handles setting up the necessary environment variables for distributed training.
#
# Basic Usage (use all available GPUs):
#   torchrun your_script_name.py [your_args]
#
# To specify the number of GPUs (e.g., 4):
#   torchrun --nproc_per_node=4 your_script_name.py [your_args]
#
# Example with arguments:
#   torchrun --nproc_per_node=7 your_script_name.py --max_epochs 500 --tv_weight 1e-5
#
# =================================================================================================

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

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
import requests
import traceback
import sys
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime
import math
import signal
import psutil

# --- Rich CLI Components for a clean UI ---
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.layout import Layout
from rich.panel import Panel

# --- Kornia for GPU-accelerated augmentations ---
import kornia.augmentation as K

# --- Configuration (Defaults) ---
DATASET_PATH = 'VisDrone2019-DET-train'
MODEL_NAME = 'yolov11n.pt' 
PATCH_SIZE = 500
BASE_LEARNING_RATE = 0.01
BASE_BATCH_SIZE = 8  # This will now be the PER-GPU batch size
DEFAULT_MAX_EPOCHS = 1000

# --- Smart Training Config ---
PLATEAU_PATIENCE = 10
EARLY_STOPPING_PATIENCE = 25 
CHECKPOINT_FILE = "patch_checkpoint.pth"
BEST_CHECKPOINT_FILE = "best_patch_checkpoint.pth"
CACHE_FILE = "preprocessed_dataset.pth"
RAM_THRESHOLD_GB = 64 # Increased threshold for powerful servers

# --- Initialize Rich Console ---
console = Console()

# --- DDP Helper Functions ---
def setup_ddp(rank, world_size):
    """Initializes the distributed process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355' # A free port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    """Cleans up the distributed process group."""
    dist.destroy_process_group()

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

# --- Top-level Functions ---
def collate_fn(batch):
    """Custom collate function to handle batches where an image might fail to load."""
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None, None
    images, boxes = zip(*batch)
    return torch.stack(images, 0), boxes

# --- Dataset Classes (with improved logging) ---
class VisDroneDatasetLazy(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_dir = os.path.join(root_dir, 'images')
        self.annotation_dir = os.path.join(root_dir, 'annotations_v11')
        self.image_files = sorted(os.listdir(self.image_dir))
        if is_main_process():
            console.print(f"✅ [green]Lazy Dataset initialized. Found {len(self.image_files)} images.[/green]")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
            original_size = image.size
        except Exception:
            return None
        boxes = []
        annotation_name = os.path.splitext(img_name)[0] + '.txt'
        annotation_path = os.path.join(self.annotation_dir, annotation_name)
        if os.path.exists(annotation_path):
            with open(annotation_path, 'r') as f:
                for line in f.readlines():
                    try:
                        parts = line.strip().split(',')
                        if len(parts) >= 4:
                            x1, y1, w, h = map(float, parts[:4])
                            boxes.append([x1, y1, x1 + w, y1 + h])
                    except (ValueError, IndexError):
                        continue
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        if boxes_tensor.nelement() > 0:
            scale_x, scale_y = 640 / original_size[0], 640 / original_size[1]
            boxes_tensor[:, [0, 2]] *= scale_x
            boxes_tensor[:, [1, 3]] *= scale_y
        return image, boxes_tensor

class VisDroneDatasetPreload(Dataset):
    def __init__(self, root_dir, transform=None):
        cache_path = os.path.join(root_dir, CACHE_FILE)
        rank = dist.get_rank()

        if is_main_process():
            if os.path.exists(cache_path):
                console.print(f"✅ [Rank {rank}] Main process found existing cache. Loading from: {cache_path}")
                try:
                    cached_data = torch.load(cache_path, map_location='cpu')
                    self.images = cached_data['images']
                    self.annotations = cached_data['annotations']
                    console.print(f"✅ [Rank {rank}] Cached dataset loaded successfully. {len(self.images)} items in memory.")
                except Exception as e:
                    console.print(f"⚠️ [Rank {rank}] Could not load cache file: {e}. Re-processing dataset.")
                    self._create_cache(root_dir, transform, cache_path)
            else:
                self._create_cache(root_dir, transform, cache_path)
        
        # All processes will wait here until the main process is done creating the cache.
        print(f"[INFO - Rank {rank}] Reached barrier, waiting for cache creation...", flush=True)
        dist.barrier()
        print(f"[INFO - Rank {rank}] Passed barrier. Cache is ready.", flush=True)

        # All processes load from the now-guaranteed-to-exist cache
        if not is_main_process():
            print(f"[INFO - Rank {rank}] Loading dataset from cache: {cache_path}", flush=True)
            cached_data = torch.load(cache_path, map_location='cpu')
            self.images = cached_data['images']
            self.annotations = cached_data['annotations']
            print(f"[INFO - Rank {rank}] Successfully loaded {len(self.images)} items from cache.", flush=True)

    def _create_cache(self, root_dir, transform, cache_path):
        """Logic for creating the cache, only run by the main process."""
        console.print(f"⏳ [magenta][Rank 0] No cache found. Starting pre-processing of dataset.[/magenta]")
        console.print(f"   [yellow]This is a one-time operation and may take a significant amount of time.[/yellow]")
        self.images = []
        self.annotations = []
        image_dir = os.path.join(root_dir, 'images')
        annotation_dir = os.path.join(root_dir, 'annotations_v11')
        image_files = sorted(os.listdir(image_dir))
        
        with Progress(console=console, disable=not is_main_process()) as progress:
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
                boxes = []
                annotation_name = f"{os.path.splitext(img_name)[0]}.txt"
                annotation_path = os.path.join(annotation_dir, annotation_name)
                if os.path.exists(annotation_path):
                    with open(annotation_path, 'r') as f:
                        for line in f.readlines():
                            try:
                                parts = line.strip().split(',')
                                if len(parts) >= 4:
                                    x1, y1, w, h = map(float, parts[:4])
                                    boxes.append([x1, y1, x1 + w, y1 + h])
                            except (ValueError, IndexError):
                                continue
                boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
                if boxes_tensor.nelement() > 0:
                    scale_x, scale_y = 640 / original_size[0], 640 / original_size[1]
                    boxes_tensor[:, [0, 2]] *= scale_x
                    boxes_tensor[:, [1, 3]] *= scale_y
                self.annotations.append(boxes_tensor)
        
        console.print(f"💾 [blue][Rank 0] Caching complete. Saving to {cache_path}...[/blue]")
        try:
            torch.save({'images': self.images, 'annotations': self.annotations}, cache_path)
            console.print(f"✅ [green][Rank 0] Dataset cached successfully.[/green]")
        except Exception as e:
            console.print(f"⚠️ [red][Rank 0] Could not save cache file: {e}[/red]")

    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        return self.images[idx], self.annotations[idx]

class DummyDataset(Dataset):
    def __init__(self, length=2048, image_size=(3, 640, 640)):
        self.length, self.image_size = length, image_size
    def __len__(self): return self.length
    def __getitem__(self, idx): return torch.rand(self.image_size)

class TotalVariationLoss(torch.nn.Module):
    def forward(self, patch):
        if patch.dim() == 3: patch = patch.unsqueeze(0)
        wh_diff = torch.sum(torch.abs(patch[:, :, :, 1:] - patch[:, :, :, :-1]))
        ww_diff = torch.sum(torch.abs(patch[:, :, 1:, :] - patch[:, :, :-1, :]))
        return (wh_diff + ww_diff) / (patch.size(2) * patch.size(3))

def autotune_batch_size(device, model, dataset_len, initial_batch_size=2):
    batch_size = initial_batch_size
    console.print(f"🚀 [magenta]Starting BATCH SIZE autotune (for a single GPU) from size {batch_size}...[/magenta]")
    while True:
        if batch_size > dataset_len:
            console.print(f"✅ [green]Batch size ({batch_size}) exceeds dataset size. Using previous valid size.[/green]")
            return batch_size // 2
        try:
            dummy_data = DummyDataset(length=batch_size)
            dataloader = DataLoader(dummy_data, batch_size=batch_size)
            images = next(iter(dataloader)).to(device)
            images.requires_grad_(True)
            with torch.amp.autocast(device_type='cuda'):
                output = model.model(images)
                dummy_loss = output[0].sum()
            dummy_loss.backward()
            console.print(f"✅ [green]Batch size {batch_size} fits in memory. Trying next size...[/green]")
            del images, dataloader, output, dummy_loss
            batch_size *= 2
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            max_size = batch_size // 2
            console.print(f"⚠️ [yellow]OOM at batch size {batch_size}. Optimal per-GPU batch size set to: {max_size}[/yellow]")
            torch.cuda.empty_cache()
            return max(1, max_size)
        except StopIteration:
            console.print(f"✅ [green]Batch size {batch_size} fits, but dataset is too small to double.[/green]")
            return batch_size

def train_adversarial_patch(rank, world_size, args, batch_size, learning_rate, log_dir):
    """Main training function, now aware of its rank and the world size."""
    if is_main_process(): cudnn.benchmark = True
    
    writer = SummaryWriter(log_dir=log_dir) if is_main_process() else None
    
    device = rank 
    model = YOLO(MODEL_NAME).to(device)
    model.model.train()

    model.model = DDP(model.model, device_ids=[device], find_unused_parameters=False)
    
    try:
        model.model = torch.compile(model.model)
        if is_main_process(): console.print("✅ [green]Model compiled successfully with torch.compile().[/green]")
    except Exception as e:
        if is_main_process(): console.print(f"⚠️ [yellow]torch.compile() failed: {e}. Running without compilation.[/yellow]")

    if args.starter_image and os.path.exists(args.starter_image):
        if is_main_process(): console.print(f"🌱 [cyan]Initializing patch from starter image: {args.starter_image}[/cyan]")
        starter_image = Image.open(args.starter_image).convert("RGB")
        transform_starter = T.Compose([T.Resize((PATCH_SIZE, PATCH_SIZE)), T.ToTensor()])
        adversarial_patch = transform_starter(starter_image).to(device, non_blocking=True)
        adversarial_patch.requires_grad_(True)
    else:
        if is_main_process(): console.print(f"🎨 [cyan]Initializing patch with random noise.[/cyan]")
        adversarial_patch = torch.rand((3, PATCH_SIZE, PATCH_SIZE), device=device, requires_grad=True)

    optimizer = torch.optim.Adam([adversarial_patch], lr=learning_rate, amsgrad=True)
    scaler = torch.amp.GradScaler(enabled=True)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=args.plateau_patience)
    total_variation = TotalVariationLoss().to(device)
    start_epoch = 0

    if args.resume and os.path.exists(args.resume):
        if is_main_process(): console.print(f"🔄 [blue]Resuming training from checkpoint: {args.resume}[/blue]")
        checkpoint = torch.load(args.resume, map_location=f'cuda:{rank}')
        model.model.module.load_state_dict(checkpoint['model_state_dict'])
        adversarial_patch.data = checkpoint['patch_state_dict'].to(device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scaler_state_dict' in checkpoint: scaler.load_state_dict(checkpoint['scaler_state_dict'])
        if 'scheduler_state_dict' in checkpoint: scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        if is_main_process(): console.print(f"Resumed from epoch {start_epoch}.")

    patch_augmentations = torch.nn.Sequential(
        K.RandomAffine(degrees=(-15, 15), scale=(0.8, 1.2), p=1.0),
        K.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.0, hue=0.0, p=1.0),
    ).to(device)

    # --- Data Loading ---
    transform = T.Compose([T.Resize((640, 640)), T.ToTensor()])
    available_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
    
    use_preload = available_ram_gb > RAM_THRESHOLD_GB
    if use_preload:
        dataset = VisDroneDatasetPreload(root_dir=DATASET_PATH, transform=transform)
    else:
        if is_main_process(): console.print(f"💾 [magenta]Using memory-safe on-demand disk loading strategy.[/magenta]")
        dataset = VisDroneDatasetLazy(root_dir=DATASET_PATH, transform=transform)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    num_workers = 0 if use_preload else min(os.cpu_count() // world_size, 16)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn, sampler=sampler, persistent_workers=True if num_workers > 0 else False)

    # --- UI Setup (Main Process Only) ---
    live = None
    if is_main_process():
        layout = Layout()
        layout.split_column(
            Layout(Panel(f"🚀 [bold magenta]Starting DDP Adversarial Patch Training[/bold magenta]\n"
                         f"   - [b]World Size[/b]: [cyan]{world_size} GPUs[/cyan]\n"
                         f"   - [b]Batch Size (per GPU)[/b]: [cyan]{batch_size}[/cyan]\n"
                         f"   - [b]Effective Batch Size[/b]: [cyan]{batch_size * world_size}[/cyan]\n"
                         f"   - [b]Scaled LR[/b]: [cyan]{learning_rate:.2e}[/cyan]\n"
                         f"   - [b]DataLoaders (per GPU)[/b]: [cyan]{num_workers}[/cyan]",
                         title="[yellow]Training Configuration[/yellow]", border_style="yellow"),
                   name="config", size=9),
            Layout(name="progress", size=3),
            Layout(name="table", ratio=1)
        )
        progress = Progress(TextColumn("[bold blue]{task.description}", justify="right"), BarColumn(bar_width=None), "[progress.percentage]{task.percentage:>3.1f}%", "•", MofNCompleteColumn(), "•", TimeRemainingColumn(), console=console)
        layout["progress"].update(Panel(progress, title="[yellow]Current Epoch Progress[/yellow]", border_style="yellow"))
        live = Live(layout, screen=True, redirect_stderr=False, vertical_overflow="visible")
        live.start()

    best_loss = float('inf')
    epochs_no_improve = 0
    epoch_results = []
    best_loss_epoch = -1

    for epoch in range(start_epoch, args.max_epochs):
        sampler.set_epoch(epoch)
        epoch_start_time = time.time()
        total_adv_loss, total_tv_loss = 0, 0
        
        task_id = progress.add_task(f"Epoch {epoch + 1}", total=len(dataloader)) if is_main_process() else None

        for i, batch_data in enumerate(dataloader):
            if batch_data is None:
                if is_main_process(): progress.update(task_id, advance=1)
                continue
            images, gt_boxes_batch = batch_data
            images = images.to(device, non_blocking=True)
            current_batch_size = images.size(0)
            
            patch_batch = adversarial_patch.unsqueeze(0).repeat(current_batch_size, 1, 1, 1)
            transformed_patch_batch = patch_augmentations(patch_batch)
            transformed_patch_batch.data.clamp_(0, 1)

            for img_idx in range(current_batch_size):
                gt_boxes = gt_boxes_batch[img_idx]
                if len(gt_boxes) > 0:
                    box = gt_boxes[random.randint(0, len(gt_boxes) - 1)].int()
                    box[0], box[1] = max(0, box[0]), max(0, box[1])
                    box[2], box[3] = min(images.shape[3] - 1, box[2]), min(images.shape[2] - 1, box[3])
                    center_x = random.randint(box[0], box[2]) if box[2] > box[0] else images.shape[3] // 2
                    center_y = random.randint(box[1], box[3]) if box[3] > box[1] else images.shape[2] // 2
                    patch_x = max(0, min(center_x - PATCH_SIZE // 2, images.shape[3] - PATCH_SIZE))
                    patch_y = max(0, min(center_y - PATCH_SIZE // 2, images.shape[2] - PATCH_SIZE))
                    images[img_idx, :, patch_y:patch_y+PATCH_SIZE, patch_x:patch_x+PATCH_SIZE] = transformed_patch_batch[img_idx]
                else:
                    x_start, y_start = random.randint(0, 640 - PATCH_SIZE), random.randint(0, 640 - PATCH_SIZE)
                    images[img_idx, :, y_start:y_start+PATCH_SIZE, x_start:x_start+PATCH_SIZE] = transformed_patch_batch[img_idx]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda'):
                raw_preds = model(images)[0].transpose(1, 2)
                adv_loss = -torch.mean(torch.max(raw_preds[..., 4:], dim=-1)[0])
                tv_loss = total_variation(adversarial_patch)
                total_loss = adv_loss + args.tv_weight * tv_loss

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            adversarial_patch.data.clamp_(0, 1)
            
            dist.all_reduce(total_loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(adv_loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(tv_loss, op=dist.ReduceOp.AVG)
            
            total_adv_loss += adv_loss.item()
            total_tv_loss += tv_loss.item()
            if is_main_process(): progress.update(task_id, advance=1)

        if is_main_process(): progress.remove_task(task_id)
        
        if is_main_process():
            avg_adv_loss = total_adv_loss / len(dataloader)
            avg_tv_loss = total_tv_loss / len(dataloader)
            avg_total_loss = avg_adv_loss + args.tv_weight * avg_tv_loss
            epoch_duration = time.time() - epoch_start_time
            current_lr = optimizer.param_groups[0]['lr']
            
            if avg_total_loss < best_loss:
                best_loss = avg_total_loss
                epochs_no_improve = 0
                best_loss_epoch = epoch + 1
                unwrapped_model = model.module.module if hasattr(model.module, 'module') else model.module
                save_dict = {
                    'epoch': epoch + 1, 
                    'model_state_dict': unwrapped_model.state_dict(),
                    'patch_state_dict': adversarial_patch.data.clone(), 
                    'optimizer_state_dict': optimizer.state_dict(), 
                    'scaler_state_dict': scaler.state_dict(), 
                    'scheduler_state_dict': scheduler.state_dict(), 
                    'batch_size': batch_size
                }
                torch.save(save_dict, os.path.join(log_dir, BEST_CHECKPOINT_FILE))
                T.ToPILImage()(adversarial_patch.cpu()).save(os.path.join(log_dir, "best_patch.png"))
            else:
                epochs_no_improve += 1
            
            epoch_results.append({"epoch": epoch + 1, "duration": epoch_duration, "adv_loss": avg_adv_loss, "tv_loss": avg_tv_loss, "total_loss": avg_total_loss, "lr": current_lr, "patience": f"{epochs_no_improve}/{args.early_stopping_patience}"})
            
            results_table = Table(title="Epoch Results", expand=True, border_style="blue")
            results_table.add_column("Epoch", justify="right", style="cyan"); results_table.add_column("Time (s)", style="magenta"); results_table.add_column("Adv Loss", style="green"); results_table.add_column("TV Loss", style="yellow"); results_table.add_column("Total Loss", style="bold red"); results_table.add_column("LR", style="cyan"); results_table.add_column("Patience", style="blue")
            for result in epoch_results[-40:]:
                row_style = "bold green" if result["epoch"] == best_loss_epoch else ""
                results_table.add_row(f"{result['epoch']}", f"{result['duration']:.2f}", f"{result['adv_loss']:.4f}", f"{result['tv_loss']:.4f}", f"{result['total_loss']:.4f}", f"{result['lr']:.1e}", result['patience'], style=row_style)
            layout["table"].update(Panel(results_table, title="[blue]Training Log (Recent Epochs)[/blue]", border_style="blue"))
            
            scheduler.step(avg_total_loss)
            writer.add_scalar('Loss/Adversarial', avg_adv_loss, epoch); writer.add_scalar('Loss/TotalVariation', avg_tv_loss, epoch); writer.add_scalar('Loss/Total', avg_total_loss, epoch); writer.add_scalar('Learning_Rate', current_lr, epoch); writer.add_image('Adversarial Patch', adversarial_patch, epoch)
            
            stop_signal = torch.tensor([0]).to(device)
            if epochs_no_improve >= args.early_stopping_patience:
                if live: live.stop()
                console.print(f"\n🛑 [bold red]Early stopping triggered after {args.early_stopping_patience} epochs with no improvement.[/bold red]")
                stop_signal = torch.tensor([1]).to(device)
            dist.broadcast(stop_signal, src=0)
            if stop_signal.item() == 1: break
        else:
            stop_signal = torch.tensor([0]).to(device)
            dist.broadcast(stop_signal, src=0)
            if stop_signal.item() == 1: break

    if is_main_process() and live: live.stop()
    if is_main_process() and writer: writer.close()


def main():
    parser = argparse.ArgumentParser(description="Distributed training of adversarial patches using DDP.")
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume a specific run.')
    parser.add_argument('--starter_image', type=str, default=None, help='Path to an image to use as the starting point for the patch.')
    parser.add_argument('--tv_weight', type=float, default=1e-4, help='Weight for the Total Variation loss term.')
    parser.add_argument('--max_epochs', type=int, default=DEFAULT_MAX_EPOCHS, help='Maximum number of epochs to train for.')
    parser.add_argument('--early_stopping_patience', type=int, default=EARLY_STOPPING_PATIENCE, help='Patience for early stopping.')
    parser.add_argument('--plateau_patience', type=int, default=PLATEAU_PATIENCE, help='Patience for LR scheduler.')
    parser.add_argument('--batch_size', type=int, default=None, help='Per-GPU batch size. If not set, will be autotuned.')

    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    setup_ddp(rank, world_size)
    print(f"[INFO - Rank {rank}/{world_size}] DDP setup complete. CUDA Device: {torch.cuda.current_device()}", flush=True)

    try:
        if is_main_process():
            parent_log_dir = "runs"
            os.makedirs(parent_log_dir, exist_ok=True)
            try:
                tb = program.TensorBoard()
                tb.configure(argv=[None, '--logdir', parent_log_dir])
                url = tb.launch()
                console.print(Panel(f"🔌 [bold]TensorBoard is running: [link={url}]{url}[/link][/bold]", title="TensorBoard", border_style="blue"))
            except Exception as e:
                console.print(f"⚠️ [yellow]Could not launch TensorBoard: {e}[/yellow]")
            if args.starter_image and args.resume: 
                console.print(f"[red]Error: --starter_image and --resume cannot be used together.[/red]"); sys.exit(1)
        
        temp_dataset_len = len(os.listdir(os.path.join(DATASET_PATH, 'images')))
        
        final_batch_size = 0
        if args.batch_size:
            final_batch_size = args.batch_size
            if is_main_process(): console.print(f"✅ [green]Using user-provided per-GPU batch size: {final_batch_size}[/green]")
        elif args.resume:
            if is_main_process():
                checkpoint = torch.load(args.resume, map_location='cpu')
                final_batch_size = checkpoint.get('batch_size', BASE_BATCH_SIZE)
                console.print(f"Resuming with batch size {final_batch_size} from checkpoint.")
        else:
            if is_main_process():
                console.print(f"🛠️ [magenta]Starting batch size autotune on Rank 0...[/magenta]")
                temp_model = YOLO(MODEL_NAME).to(rank)
                temp_model.model.train()
                final_batch_size = autotune_batch_size(device=rank, model=temp_model.model, dataset_len=temp_dataset_len)
                del temp_model
                torch.cuda.empty_cache()
        
        print(f"[INFO - Rank {rank}] Waiting to receive batch size...", flush=True)
        batch_size_tensor = torch.tensor([final_batch_size], dtype=torch.int64).to(rank)
        dist.broadcast(batch_size_tensor, src=0)
        final_batch_size = batch_size_tensor.item()
        print(f"[INFO - Rank {rank}] Received batch size: {final_batch_size}", flush=True)

        effective_batch_size = final_batch_size * world_size
        scaled_lr = BASE_LEARNING_RATE * math.sqrt(effective_batch_size / BASE_BATCH_SIZE)
        
        log_dir = None
        if is_main_process():
            session_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            log_dir = os.path.join("runs", session_timestamp)
            if args.resume: log_dir = os.path.dirname(args.resume)
            os.makedirs(log_dir, exist_ok=True)
        
        log_dir_list = [log_dir]
        dist.broadcast_object_list(log_dir_list, src=0)
        log_dir = log_dir_list[0]

        if is_main_process():
            console.print("✅ [green]Setup complete. Starting training loop...[/green]")

        train_adversarial_patch(rank, world_size, args, final_batch_size, scaled_lr, log_dir)

    except Exception as e:
        if is_main_process():
            console.print("\n" + "="*60)
            console.print(f"💥 [bold red]An unexpected error occurred![/bold red]")
            console.print("="*60)
            error_info = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            console.print(error_info)
        print(f"[ERROR - Rank {rank}] Encountered an exception: {e}", flush=True)
        traceback.print_exc()
    finally:
        print(f"[INFO - Rank {rank}] Cleaning up DDP.", flush=True)
        cleanup_ddp()

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True) 
    main()
