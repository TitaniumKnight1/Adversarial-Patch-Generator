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
import requests
import traceback
import sys
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime
import math
import signal
import torch.multiprocessing
import psutil

# --- Rich CLI Components for a clean UI ---
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.layout import Layout
from rich.panel import Panel

# --- Configuration (Defaults) ---
DATASET_PATH = 'VisDrone2019-DET-train'
MODEL_NAME = 'yolov11n.pt' 
PATCH_SIZE = 500
BASE_LEARNING_RATE = 0.01
BASE_BATCH_SIZE = 8
DEFAULT_MAX_EPOCHS = 1000

# --- Smart Training Config ---
PLATEAU_PATIENCE = 10
EARLY_STOPPING_PATIENCE = 25 
CHECKPOINT_FILE = "patch_checkpoint.pth"
BEST_CHECKPOINT_FILE = "best_patch_checkpoint.pth"
CACHE_FILE = "preprocessed_dataset.pth"
RAM_THRESHOLD_GB = 32

# --- Initialize Rich Console ---
console = Console()

# --- Top-level Functions ---
def collate_fn(batch):
    """Custom collate function to handle batches where an image might fail to load."""
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None, None
    images, boxes = zip(*batch)
    return torch.stack(images, 0), boxes

# --- Dataset Classes ---
class VisDroneDatasetLazy(Dataset):
    """Loads images and annotations from disk on-the-fly."""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_dir = os.path.join(root_dir, 'images')
        self.annotation_dir = os.path.join(root_dir, 'annotations_v11')
        self.image_files = sorted(os.listdir(self.image_dir))
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
    """Loads and pre-processes the entire dataset into RAM."""
    def __init__(self, root_dir, transform=None):
        cache_path = os.path.join(root_dir, CACHE_FILE)
        if os.path.exists(cache_path):
            console.print(f"⏳ [blue]Loading pre-processed dataset from cache: {cache_path}[/blue]")
            try:
                cached_data = torch.load(cache_path, weights_only=False)
                self.images = cached_data['images']
                self.annotations = cached_data['annotations']
                console.print(f"✅ [green]Cached dataset loaded successfully. {len(self.images)} items in memory.[/green]")
                return
            except Exception as e:
                console.print(f"⚠️ [yellow]Could not load cache file: {e}. Re-processing dataset.[/yellow]")

        console.print(f"⏳ [magenta]No cache found. Pre-processing dataset into tensors... This may take a moment.[/magenta]")
        self.images = []
        self.annotations = []
        image_dir = os.path.join(root_dir, 'images')
        annotation_dir = os.path.join(root_dir, 'annotations_v11')
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
        
        console.print(f"💾 [blue]Saving pre-processed dataset to cache for future runs...[/blue]")
        try:
            torch.save({'images': self.images, 'annotations': self.annotations}, cache_path)
            console.print(f"✅ [green]Dataset cached successfully at: {cache_path}[/green]")
        except Exception as e:
            console.print(f"⚠️ [red]Could not save cache file: {e}[/red]")

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
    console.print(f"🚀 [magenta]Starting ACCURATE batch size autotune from size {batch_size}...[/magenta]")
    console.print(f"   [yellow]This test simulates a full forward/backward pass for accuracy.[/yellow]")
    while True:
        if batch_size > dataset_len:
            console.print(f"✅ [green]Batch size ({batch_size}) exceeds dataset size ({dataset_len}). Using previous valid size.[/green]")
            return batch_size // 2
        try:
            dummy_data = DummyDataset(length=batch_size)
            dataloader = DataLoader(dummy_data, batch_size=batch_size)
            images = next(iter(dataloader)).to(device)
            images.requires_grad_(True)
            with torch.amp.autocast(device.type):
                output = model.model(images)
                dummy_loss = output[0].sum()
            dummy_loss.backward()
            console.print(f"✅ [green]Batch size {batch_size} (full pass) fits in memory. Trying next size...[/green]")
            del images, dataloader, output, dummy_loss
            batch_size *= 2
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            max_size = batch_size // 2
            console.print(f"⚠️ [yellow]OOM at batch size {batch_size}. Optimal batch size set to: {max_size}[/yellow]")
            torch.cuda.empty_cache()
            return max(1, max_size)
        except StopIteration:
            console.print(f"✅ [green]Batch size {batch_size} fits, but dataset is too small to double.[/green]")
            return batch_size

def train_adversarial_patch(batch_size, learning_rate, log_dir, max_epochs, device, tv_weight, dataloader, num_workers, early_stopping_patience, pin_memory, resume_path=None, starter_image_path=None):
    if device.type == 'cuda': cudnn.benchmark = True
    writer = SummaryWriter(log_dir=log_dir)
    model = YOLO(MODEL_NAME).to(device)
    if isinstance(model.model, torch.nn.DataParallel):
        console.print(f"🚀 [magenta]Using DataParallel on {torch.cuda.device_count()} specified GPUs![/magenta]")
    model.model.train()
    if device.type == 'cuda':
        try:
            if not isinstance(model.model, torch.nn.DataParallel):
                model.model = torch.compile(model.model)
                console.print("✅ [green]Model compiled successfully with torch.compile().[/green]")
        except Exception as e:
            console.print(f"⚠️ [yellow]torch.compile() failed: {e}. Running without compilation.[/yellow]")

    if starter_image_path and os.path.exists(starter_image_path):
        console.print(f"🌱 [cyan]Initializing patch from starter image: {starter_image_path}[/cyan]")
        starter_image = Image.open(starter_image_path).convert("RGB")
        transform_starter = T.Compose([T.Resize((PATCH_SIZE, PATCH_SIZE)), T.ToTensor()])
        adversarial_patch = transform_starter(starter_image).to(device, non_blocking=True)
        adversarial_patch.requires_grad_(True)
    else:
        console.print(f"🎨 [cyan]Initializing patch with random noise.[/cyan]")
        adversarial_patch = torch.rand((3, PATCH_SIZE, PATCH_SIZE), device=device, requires_grad=True)

    optimizer = torch.optim.Adam([adversarial_patch], lr=learning_rate, amsgrad=True)
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=PLATEAU_PATIENCE)
    total_variation = TotalVariationLoss().to(device)
    start_epoch = 0
    if resume_path and os.path.exists(resume_path):
        console.print(f"🔄 [blue]Resuming training from checkpoint: {resume_path}[/blue]")
        checkpoint = torch.load(resume_path)
        adversarial_patch.data = checkpoint['patch_state_dict'].to(device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scaler_state_dict' in checkpoint: scaler.load_state_dict(checkpoint['scaler_state_dict'])
        if 'scheduler_state_dict' in checkpoint: scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        console.print(f"Resumed from epoch {start_epoch}.")

    # --- Rich UI Layout ---
    layout = Layout()
    layout.split(
        Layout(name="header", size=3),
        Layout(ratio=1, name="main"),
        Layout(size=10, name="footer")
    )
    # FIX: Give the side panel more space (40% of width)
    layout["main"].split_row(Layout(name="side", ratio=2), Layout(name="body", ratio=3))
    layout["footer"].update(Panel(f"🚀 [bold magenta]Starting Adversarial Patch Training[/bold magenta]\n"
                                 f"   - [b]Device[/b]: [cyan]{device.type.upper()}[/cyan]\n"
                                 f"   - [b]Batch Size[/b]: [cyan]{batch_size}[/cyan]\n"
                                 f"   - [b]Scaled LR[/b]: [cyan]{learning_rate:.2e}[/cyan]\n"
                                 f"   - [b]DataLoaders[/b]: [cyan]{num_workers}[/cyan]\n"
                                 f"   - [b]TV Weight[/b]: [cyan]{tv_weight}[/cyan]\n"
                                 f"   - [b]Patience[/b]: [cyan]{early_stopping_patience}[/cyan]",
                                 title="[yellow]Training Configuration[/yellow]", border_style="yellow"))
    
    progress = Progress(
        TextColumn("[bold blue]{task.description}", justify="right"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%", "•",
        MofNCompleteColumn(), "•",
        TimeRemainingColumn(),
        console=console
    )
    layout["side"].update(Panel(progress, title="[yellow]Epoch Progress[/yellow]", border_style="yellow"))

    best_loss = float('inf')
    epochs_no_improve = 0
    
    # FIX: Store epoch results to rebuild table for highlighting
    epoch_results = []
    best_loss_epoch = -1

    with Live(layout, screen=True, redirect_stderr=False, vertical_overflow="visible") as live:
        for epoch in range(start_epoch, max_epochs):
            epoch_start_time = time.time()
            total_adv_loss, total_tv_loss = 0, 0
            
            task_id = progress.add_task(f"Epoch {epoch + 1}", total=len(dataloader))

            for i, batch_data in enumerate(dataloader):
                if batch_data is None:
                    progress.update(task_id, advance=1)
                    continue
                images, gt_boxes_batch = batch_data
                images = images.to(device, non_blocking=pin_memory)
                
                # --- Patch Transformation ---
                angle, scale = random.uniform(-15, 15), random.uniform(0.8, 1.2)
                transformed_patch = TF.rotate(adversarial_patch, angle)
                transformed_patch = TF.resize(transformed_patch, (int(PATCH_SIZE * scale), int(PATCH_SIZE * scale)))
                transformed_patch = TF.center_crop(transformed_patch, (PATCH_SIZE, PATCH_SIZE))
                transformed_patch = TF.adjust_brightness(transformed_patch, random.uniform(0.7, 1.3))
                transformed_patch = TF.adjust_contrast(transformed_patch, random.uniform(0.7, 1.3))
                transformed_patch.data.clamp_(0,1)

                # --- Patch Application ---
                for img_idx in range(images.size(0)):
                    gt_boxes = gt_boxes_batch[img_idx]
                    if len(gt_boxes) > 0:
                        box = gt_boxes[random.randint(0, len(gt_boxes) - 1)].int()
                        center_x, center_y = random.randint(box[0], box[2]), random.randint(box[1], box[3])
                        patch_x = max(0, min(center_x - PATCH_SIZE // 2, images.shape[3] - PATCH_SIZE))
                        patch_y = max(0, min(center_y - PATCH_SIZE // 2, images.shape[2] - PATCH_SIZE))
                        images[img_idx, :, patch_y:patch_y+PATCH_SIZE, patch_x:patch_x+PATCH_SIZE] = transformed_patch
                    else:
                        x_start, y_start = random.randint(0, 640 - PATCH_SIZE), random.randint(0, 640 - PATCH_SIZE)
                        images[img_idx, :, y_start:y_start+PATCH_SIZE, x_start:x_start+PATCH_SIZE] = transformed_patch

                # --- Training Step ---
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device.type):
                    raw_preds = model.model(images)[0].transpose(1, 2)
                    adv_loss = -torch.mean(torch.max(raw_preds[..., 4:], dim=-1)[0])
                    tv_loss = total_variation(adversarial_patch)
                    total_loss = adv_loss + tv_weight * tv_loss

                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                adversarial_patch.data.clamp_(0, 1)
                
                total_adv_loss += adv_loss.item()
                total_tv_loss += tv_loss.item()
                progress.update(task_id, advance=1)

            progress.remove_task(task_id)
            
            # --- Epoch Summary ---
            avg_adv_loss = total_adv_loss / len(dataloader) if len(dataloader) > 0 else 0
            avg_tv_loss = total_tv_loss / len(dataloader) if len(dataloader) > 0 else 0
            avg_total_loss = avg_adv_loss + tv_weight * avg_tv_loss
            epoch_duration = time.time() - epoch_start_time
            current_lr = optimizer.param_groups[0]['lr']
            
            if avg_total_loss < best_loss:
                best_loss = avg_total_loss
                epochs_no_improve = 0
                best_loss_epoch = epoch + 1
                torch.save({'epoch': epoch + 1, 'patch_state_dict': adversarial_patch.data.clone(), 'optimizer_state_dict': optimizer.state_dict(), 'scaler_state_dict': scaler.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'batch_size': batch_size}, os.path.join(log_dir, BEST_CHECKPOINT_FILE))
                patch_image = T.ToPILImage()(adversarial_patch.cpu())
                patch_image.save(os.path.join(log_dir, "best_patch.png"))
            else:
                epochs_no_improve += 1
            
            # Store results of the current epoch
            epoch_results.append({
                "epoch": epoch + 1, "duration": epoch_duration, "adv_loss": avg_adv_loss,
                "tv_loss": avg_tv_loss, "total_loss": avg_total_loss, "lr": current_lr,
                "patience": f"{epochs_no_improve}/{early_stopping_patience}"
            })

            # Rebuild the results table to update highlighting
            results_table = Table(title="Epoch Results", expand=True)
            results_table.add_column("Epoch", justify="right", style="cyan")
            results_table.add_column("Time (s)", style="magenta")
            results_table.add_column("Adv Loss", style="green")
            results_table.add_column("TV Loss", style="yellow")
            results_table.add_column("Total Loss", style="bold red")
            results_table.add_column("LR", style="cyan")
            results_table.add_column("Patience", style="blue")

            for result in epoch_results:
                row_style = "bold green" if result["epoch"] == best_loss_epoch else ""
                results_table.add_row(
                    f"{result['epoch']}", f"{result['duration']:.2f}", f"{result['adv_loss']:.4f}",
                    f"{result['tv_loss']:.4f}", f"{result['total_loss']:.4f}", f"{result['lr']:.1e}",
                    result['patience'], style=row_style
                )
            layout["body"].update(results_table)
            
            scheduler.step(avg_total_loss)
            writer.add_scalar('Loss/Adversarial', avg_adv_loss, epoch)
            writer.add_scalar('Loss/TotalVariation', avg_tv_loss, epoch)
            writer.add_scalar('Loss/Total', avg_total_loss, epoch)
            writer.add_scalar('Learning_Rate', current_lr, epoch)
            writer.add_image('Adversarial Patch', adversarial_patch, epoch)
            
            torch.save({'epoch': epoch + 1, 'patch_state_dict': adversarial_patch.data.clone(), 'optimizer_state_dict': optimizer.state_dict(), 'scaler_state_dict': scaler.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'batch_size': batch_size}, os.path.join(log_dir, CHECKPOINT_FILE))

            if epochs_no_improve >= early_stopping_patience:
                live.stop()
                console.print(f"\n🛑 [bold red]Early stopping triggered after {early_stopping_patience} epochs with no improvement.[/bold red]")
                console.print(f"   Best loss achieved: {best_loss:.4f}")
                break
    
    writer.close()
    console.print(f"\n🎉 [bold green]Training run finished.[/bold green]")


def send_crash_notification(error_message):
    try:
        requests.post("https://ntfy.sh/PatchTraining", data=error_message.encode('utf-8'), headers={"Title": "Patch Training Script CRASHED", "Priority": "high", "Tags": "rotating_light,x"})
        console.print(f"\n🚨 [red]Crash notification sent to ntfy.[/red]")
    except Exception as e:
        console.print(f"Failed to send crash notification: {e}")

def signal_handler(sig, frame):
    console.print(f'\n[bold red]Ctrl+C detected! Exiting gracefully...[/bold red]')
    sys.exit(0)

if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.multiprocessing.set_start_method('spawn', force=True)
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser(description="Train adversarial patches against a YOLO model with automatic LR scaling and early stopping.")
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume a specific run.')
    parser.add_argument('--starter_image', type=str, default=None, help='Path to an image to use as the starting point for the patch.')
    parser.add_argument('--tv_weight', type=float, default=1e-4, help='Weight for the Total Variation loss term.')
    parser.add_argument('--max_epochs', type=int, default=DEFAULT_MAX_EPOCHS, help='Maximum number of epochs to train for.')
    parser.add_argument('--early_stopping_patience', type=int, default=EARLY_STOPPING_PATIENCE, help='Number of epochs with no improvement to wait before stopping.')
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=None, help='Specific GPU IDs to use (e.g., 0 1 2). If not specified, uses single GPU. If multiple IDs are given, enables DataParallel.')
    args = parser.parse_args()

    if args.gpu_ids:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpu_ids))
        console.print(f"✅ [green]Running on specified GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}[/green]")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        console.print(f"⚠️ [yellow]CUDA not available. Running on CPU. This will be very slow.[/yellow]")

    parent_log_dir = "runs"
    os.makedirs(parent_log_dir, exist_ok=True)
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', parent_log_dir])
    url = tb.launch()
    console.print(Panel(f"🔌 [bold]TensorBoard is running: [link={url}]{url}[/link][/bold]", title="TensorBoard", border_style="blue"))

    if args.starter_image and args.resume: 
        console.print(f"[red]Error: --starter_image and --resume cannot be used together.[/red]"); sys.exit(1)
    
    temp_dataset_len = len(os.listdir(os.path.join(DATASET_PATH, 'images')))

    if args.resume:
        checkpoint = torch.load(args.resume)
        final_batch_size = checkpoint.get('batch_size', BASE_BATCH_SIZE)
        console.print(f"Resuming with batch size {final_batch_size} from checkpoint.")
    elif device.type == 'cuda':
        console.print(f"🛠️ [magenta]Preparing for autotune...[/magenta]")
        temp_model = YOLO(MODEL_NAME).to(device)
        if args.gpu_ids and len(args.gpu_ids) > 1:
            temp_model.model = torch.nn.DataParallel(temp_model.model)
        temp_model.model.train()
        final_batch_size = autotune_batch_size(device=device, model=temp_model, dataset_len=temp_dataset_len)
        del temp_model
        torch.cuda.empty_cache()
    else:
        console.print("Autotune is only for CUDA. Using base batch size.")
        final_batch_size = BASE_BATCH_SIZE

    num_gpus = len(args.gpu_ids) if args.gpu_ids else 1
    effective_batch_size = final_batch_size * num_gpus
    scaled_lr = BASE_LEARNING_RATE * (effective_batch_size / BASE_BATCH_SIZE)
    
    session_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(parent_log_dir, session_timestamp)
    if args.resume:
        log_dir = os.path.dirname(args.resume)

    try:
        if not os.path.exists(DATASET_PATH):
            console.print(f"[red]Error: Dataset path not found: '{DATASET_PATH}'[/red]"); sys.exit(1)
        
        transform = T.Compose([T.Resize((640, 640)), T.ToTensor()])
        available_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
        console.print(f"ℹ️ Detected {available_ram_gb:.2f} GB of available system RAM.")

        if available_ram_gb > RAM_THRESHOLD_GB:
            console.print(f"🚀 [magenta]High RAM detected. Using high-performance RAM pre-loading strategy.[/magenta]")
            dataset = VisDroneDatasetPreload(root_dir=DATASET_PATH, transform=transform)
            num_workers = 0 
        else:
            console.print(f"💾 [magenta]Using memory-safe on-demand disk loading strategy.[/magenta]")
            dataset = VisDroneDatasetLazy(root_dir=DATASET_PATH, transform=transform)
            num_workers = min(os.cpu_count(), 16)
        
        pin_memory = (device.type == 'cuda')
        dataloader = DataLoader(dataset, batch_size=final_batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn, persistent_workers=True if num_workers > 0 else False)

        train_adversarial_patch(
            batch_size=final_batch_size, 
            learning_rate=scaled_lr,
            log_dir=log_dir,
            max_epochs=args.max_epochs,
            device=device,
            resume_path=args.resume,
            starter_image_path=args.starter_image,
            tv_weight=args.tv_weight,
            dataloader=dataloader,
            num_workers=num_workers,
            early_stopping_patience=args.early_stopping_patience,
            pin_memory=pin_memory
        )
    except Exception as e:
        console.print("\n" + "="*60)
        console.print(f"💥 [bold red]An unexpected error occurred! Sending notification...[/bold red]")
        console.print("="*60)
        error_info = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        send_crash_notification(error_info)
        raise e
