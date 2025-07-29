import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from ultralytics import YOLO
from PIL import Image
import os
import random
import argparse
import math
import matplotlib.pyplot as plt
from rich.console import Console
from rich.progress import Progress

# --- Configuration ---
MODEL_NAME = 'yolov11n.pt' 
PATCH_SIZE = 100
DEFAULT_DATASET_PATH = 'VisDrone2019-DET-train'

# --- Initialize Rich Console ---
console = Console()

# --- Data Loading (Simplified for LR Finder) ---
def collate_fn(batch):
    """Custom collate function to handle batches where an image might fail to load."""
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None, None
    images, _ = zip(*batch) # We don't need boxes for the LR finder
    return torch.stack(images, 0)

class VisDroneDatasetLazy(Dataset):
    """A lightweight version of the dataset for the LR finder."""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_dir = os.path.join(root_dir, 'images')
        self.image_files = sorted(os.listdir(self.image_dir))
        console.print(f"✅ [green]Dataset initialized. Found {len(self.image_files)} images.[/green]")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            return None
        
        if self.transform:
            image = self.transform(image)
        
        # Return a dummy tensor for the boxes placeholder
        return image, torch.empty(0)

# --- Learning Rate Finder Function ---
def find_learning_rate(model, dataloader, device, start_lr=1e-7, end_lr=1, num_iter=200, beta=0.98):
    """
    Implements the learning rate finder with loss smoothing.
    Increases LR exponentially and records a smoothed loss to find the optimal starting point.
    """
    console.print("🚀 [bold magenta]Starting Learning Rate Finder...[/bold magenta]")
    model.model.eval()
    
    adversarial_patch = torch.rand((3, PATCH_SIZE, PATCH_SIZE), device=device, requires_grad=True)
    optimizer = torch.optim.AdamW([adversarial_patch], lr=start_lr)
    
    lr_lambda = lambda x: math.exp(x * math.log(end_lr / start_lr) / (num_iter - 1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    lrs = []
    losses = []
    avg_loss = 0.0
    
    iterator = iter(dataloader)
    with Progress(console=console) as progress:
        task = progress.add_task("[cyan]Running LR Finder...", total=num_iter)
        for i in range(num_iter):
            try:
                images = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                images = next(iterator)
            
            if images is None:
                continue

            images = images.to(device)
            
            x_start, y_start = random.randint(0, 640 - PATCH_SIZE), random.randint(0, 640 - PATCH_SIZE)
            images[:, :, y_start:y_start+PATCH_SIZE, x_start:x_start+PATCH_SIZE] = adversarial_patch

            optimizer.zero_grad()
            
            raw_preds = model.model(images)[0].transpose(1, 2)
            loss = torch.mean(torch.max(raw_preds[..., 4:], dim=-1)[0])

            # MODIFICATION: Implement Exponential Moving Average for loss smoothing
            avg_loss = beta * avg_loss + (1 - beta) * loss.item()
            smoothed_loss = avg_loss / (1 - beta**(i + 1))

            if torch.isnan(loss):
                console.print("[bold red]Loss exploded. Stopping LR finder.[/bold red]")
                break

            (-loss).backward()
            optimizer.step()
            
            lrs.append(optimizer.param_groups[0]['lr'])
            losses.append(smoothed_loss) # Append the smoothed loss for plotting
            
            scheduler.step()
            progress.update(task, advance=1, description=f"[cyan]Running LR Finder... (Smoothed Loss: {smoothed_loss:.4f})")
            

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(lrs, losses)
    plt.xscale("log")
    plt.xlabel("Learning Rate (log scale)")
    plt.ylabel("Smoothed Adversarial Loss")
    plt.title("Learning Rate Finder (Smoothed)")
    plt.grid(True, which="both", ls="--")
    plt.savefig("lr_finder.png")
    console.print("✅ [bold green]LR finder plot saved to 'lr_finder.png'.[/bold green]")
    console.print("➡️ [bold cyan]Analyze the plot: Choose a learning rate from the steepest upward slope before the loss plateaus or explodes.[/bold cyan]")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Learning Rate Finder for Adversarial Patch Training.")
    parser.add_argument('--dataset_path', type=str, default=DEFAULT_DATASET_PATH, help='Path to the training dataset.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size to use for the LR finder.')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        console.print(f"⚠️ [yellow]CUDA not available. Running on CPU.[/yellow]")

    console.print("🛠️ [magenta]Setting up model and dataloader...[/magenta]")
    model = YOLO(MODEL_NAME).to(device)
    
    transform = T.Compose([T.Resize((640, 640)), T.ToTensor()])
    dataset = VisDroneDatasetLazy(root_dir=args.dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    find_learning_rate(model, dataloader, device)
