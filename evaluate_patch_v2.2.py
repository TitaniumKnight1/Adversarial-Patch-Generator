import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import os
import random
import time
import argparse
import numpy as np
import sys
import math

# --- Rich CLI Components for a clean UI ---
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.panel import Panel

# --- Configuration ---
DEFAULT_MODEL_NAME = 'yolov11n.pt'
DEFAULT_DATASET_PATH = 'VisDrone2019-DET-val' # Use a validation/test set for evaluation
DEFAULT_PATCH_PATH = 'runs/best_patch.png'

# --- Evaluation Parameters ---
CONF_THRESHOLD = 0.25  # Confidence threshold for an object to be considered "detected"
IOU_THRESHOLD = 0.5    # IoU threshold to match a detection after patching

# --- Initialize Rich Console ---
console = Console()

def calculate_iou(box1, box2):
    """Calculates Intersection over Union (IoU) for two bounding boxes."""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area if union_area > 0 else 0
    return iou

def draw_boxes(image, boxes, confidences, color="lime", width=2):
    """Draws bounding boxes and confidences on a PIL image."""
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    for box, conf in zip(boxes, confidences):
        draw.rectangle(box.tolist(), outline=color, width=width)
        text = f"{conf:.2f}"
        text_bbox = draw.textbbox((box[0], box[1] - 15), text, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((box[0], box[1] - 15), text, fill="black", font=font)
    return image

class EvaluationDataset(Dataset):
    """
    Loads images from a directory for evaluation.
    This is a "lazy" loader that reads from disk on-the-fly.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_dir = os.path.join(root_dir, 'images')
        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"Image directory not found at: {self.image_dir}")
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        console.print(f"✅ [green]Evaluation dataset initialized. Found {len(self.image_files)} images.[/green]")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            console.print(f"⚠️ [yellow]Could not load image {img_path}. Skipping. Error: {e}[/yellow]")
            return None, None

        if self.transform:
            image = self.transform(image)
            
        return image, img_name

def collate_fn_eval(batch):
    """Custom collate function to handle None values from failed image loads."""
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch:
        return None, None
    images, names = zip(*batch)
    return torch.stack(images, 0), names


def evaluate_patch(args):
    """Main function to run the adversarial patch evaluation."""
    # --- Setup ---
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() and args.gpu_id is not None else "cpu")
    if device.type == 'cpu':
        console.print("⚠️ [yellow]CUDA not available or GPU not specified. Running on CPU.[/yellow]")
    else:
        console.print(f"🚀 [magenta]Running evaluation on device: {device}[/magenta]")

    # --- Load Model ---
    console.print(f"⏳ [blue]Loading YOLO model: {args.model_name}[/blue]")
    model = YOLO(args.model_name).to(device)
    model.model.eval()

    # --- Load Patch ---
    console.print(f"🎨 [blue]Loading adversarial patch from: {args.patch_path}[/blue]")
    if not os.path.exists(args.patch_path):
        console.print(f"❌ [red]Error: Patch file not found at {args.patch_path}[/red]")
        sys.exit(1)
    patch_image = Image.open(args.patch_path).convert("RGB")
    patch_tensor = T.ToTensor()(patch_image).to(device)

    # --- Load Data ---
    transform = T.Compose([T.Resize((640, 640)), T.ToTensor()])
    dataset = EvaluationDataset(root_dir=args.dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn_eval)
    
    # --- Output Directory ---
    if args.save_examples > 0:
        output_dir = "evaluation_results_v2"
        os.makedirs(output_dir, exist_ok=True)
        console.print(f"🖼️  [cyan]Saving visual examples to '{output_dir}/'[/cyan]")

    # --- Evaluation Counters ---
    total_targets = 0
    successful_attacks = 0
    total_clean_detections = 0
    total_patched_detections = 0
    confidence_before = []
    confidence_after = []

    # --- Rich UI ---
    progress = Progress(
        TextColumn("[bold blue]{task.description}", justify="right"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%", "•",
        MofNCompleteColumn(), "•",
        TimeRemainingColumn(),
        console=console
    )
    
    with progress:
        task_id = progress.add_task("Evaluating Patch", total=len(dataloader))
        
        for images, img_names in dataloader:
            if images is None:
                progress.update(task_id, advance=1)
                continue
            
            image_tensor = images.to(device)
            img_name = img_names[0]

            # --- 1. Get Detections on Clean Image ---
            with torch.no_grad():
                results_clean = model(image_tensor, verbose=False, conf=args.conf_thresh)
            
            clean_boxes = results_clean[0].boxes.xyxy
            clean_confs = results_clean[0].boxes.conf
            total_clean_detections += len(clean_boxes)

            if len(clean_boxes) == 0:
                progress.update(task_id, advance=1)
                continue

            # --- 2. Select Target and Apply Patch ---
            target_idx = torch.argmax(clean_confs)
            target_box = clean_boxes[target_idx]
            target_conf = clean_confs[target_idx]
            
            total_targets += 1
            confidence_before.append(target_conf.item())

            # --- UPDATED: DYNAMIC PATCH SCALING BASED ON AREA ---
            # Calculate target area
            target_w = target_box[2] - target_box[0]
            target_h = target_box[3] - target_box[1]
            target_area = target_w * target_h

            # Dynamically scale patch area to be a fraction of the target's area
            patch_area = target_area * args.patch_scale
            patch_base_size = int(math.sqrt(patch_area)) # Patch is square
            patch_base_size = max(patch_base_size, 16) # Enforce a minimum pixel size

            # Apply random transformations (rotation, brightness, etc.)
            angle = random.uniform(-15, 15)
            transformed_patch = TF.rotate(patch_tensor, angle)
            transformed_patch = TF.resize(transformed_patch, (patch_base_size, patch_base_size))
            transformed_patch = TF.adjust_brightness(transformed_patch, random.uniform(0.8, 1.2))
            transformed_patch = TF.adjust_contrast(transformed_patch, random.uniform(0.8, 1.2))
            transformed_patch.data.clamp_(0,1)

            # Place patch on a random edge of the target box to ensure partial occlusion
            patched_image = image_tensor.clone().squeeze(0)
            side = random.randint(0, 3) # 0: top, 1: right, 2: bottom, 3: left
            
            # Use occlusion_level to control overlap. 0.0=touching, 1.0=centered on edge
            occlusion = args.occlusion_level * (patch_base_size / 2)

            if side == 0: # Top edge
                center_x = int(random.uniform(target_box[0], target_box[2]))
                center_y = int(target_box[1] + occlusion)
            elif side == 1: # Right edge
                center_x = int(target_box[2] - occlusion)
                center_y = int(random.uniform(target_box[1], target_box[3]))
            elif side == 2: # Bottom edge
                center_x = int(random.uniform(target_box[0], target_box[2]))
                center_y = int(target_box[3] - occlusion)
            else: # Left edge (side == 3)
                center_x = int(target_box[0] + occlusion)
                center_y = int(random.uniform(target_box[1], target_box[3]))

            # Calculate top-left corner for pasting, ensuring it's within image bounds
            patch_x = int(center_x - patch_base_size / 2)
            patch_y = int(center_y - patch_base_size / 2)
            
            x1 = max(0, patch_x)
            y1 = max(0, patch_y)
            x2 = min(640, patch_x + patch_base_size)
            y2 = min(640, patch_y + patch_base_size)

            # Apply the patch slice by slice to handle boundary conditions
            patched_image[:, y1:y2, x1:x2] = transformed_patch[
                :, 
                max(0, y1-patch_y):min(patch_base_size, y2-patch_y), 
                max(0, x1-patch_x):min(patch_base_size, x2-patch_x)
            ]
            patched_image_tensor = patched_image.unsqueeze(0)
            # --- END OF UPDATED LOGIC ---

            # --- 3. Get Detections on Patched Image ---
            with torch.no_grad():
                results_patched = model(patched_image_tensor, verbose=False, conf=args.conf_thresh)
            
            patched_boxes = results_patched[0].boxes.xyxy
            patched_confs = results_patched[0].boxes.conf
            total_patched_detections += len(patched_boxes)

            # --- 4. Check if Attack was Successful ---
            best_iou = 0
            best_match_conf = 0
            if len(patched_boxes) > 0:
                for p_box, p_conf in zip(patched_boxes, patched_confs):
                    iou = calculate_iou(target_box.cpu().numpy(), p_box.cpu().numpy())
                    if iou > best_iou:
                        best_iou = iou
                        best_match_conf = p_conf.item()

            if best_iou < args.iou_thresh:
                successful_attacks += 1
                confidence_after.append(0) # Vanished
            else:
                confidence_after.append(best_match_conf) # Not vanished

            # --- 5. Save Visual Example if Requested ---
            if total_targets <= args.save_examples:
                clean_pil = T.ToPILImage()(image_tensor.cpu().squeeze(0))
                patched_pil = T.ToPILImage()(patched_image_tensor.cpu().squeeze(0))

                clean_pil = draw_boxes(clean_pil, clean_boxes.cpu(), clean_confs.cpu(), color="lime")
                draw = ImageDraw.Draw(clean_pil)
                draw.rectangle(target_box.cpu().tolist(), outline="yellow", width=4)

                patched_pil = draw_boxes(patched_pil, patched_boxes.cpu(), patched_confs.cpu(), color="red")

                combined_img = Image.new('RGB', (clean_pil.width * 2 + 10, clean_pil.height), "grey")
                combined_img.paste(clean_pil, (0, 0))
                combined_img.paste(patched_pil, (clean_pil.width + 10, 0))
                
                save_path = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_attack.jpg")
                combined_img.save(save_path)

            progress.update(task_id, advance=1)

    # --- Final Report ---
    console.print("\n" + "="*60)
    console.print("📊 [bold green]Evaluation Complete[/bold green]")
    console.print("="*60)
    
    asr = (successful_attacks / total_targets) * 100 if total_targets > 0 else 0
    avg_conf_before = np.mean(confidence_before) if confidence_before else 0
    avg_conf_after = np.mean([c for c in confidence_after if c > 0]) if any(c > 0 for c in confidence_after) else 0
    
    table = Table(title="Adversarial Patch Evaluation Results", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="yellow")

    table.add_row("Total Images Processed", f"{len(dataloader)}")
    table.add_row("Total Objects Targeted", f"{total_targets}")
    table.add_row("Successful Attacks (IoU < threshold)", f"{successful_attacks}")
    table.add_row("[bold]Attack Success Rate (ASR)[/bold]", f"[bold red]{asr:.2f}%[/bold red]")
    table.add_row("-" * 20, "-" * 20)
    table.add_row("Avg Confidence (Before Attack)", f"{avg_conf_before:.3f}")
    table.add_row("Avg Confidence (After Attack, if detected)", f"{avg_conf_after:.3f}")
    table.add_row("Total Detections (Clean)", f"{total_clean_detections}")
    table.add_row("Total Detections (Patched)", f"{total_patched_detections}")

    console.print(table)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate an adversarial patch against a YOLO model with realistic placement.")
    parser.add_argument('--patch_path', type=str, default=DEFAULT_PATCH_PATH, help='Path to the adversarial patch image file.')
    parser.add_argument('--dataset_path', type=str, default=DEFAULT_DATASET_PATH, help='Path to the root of the evaluation dataset.')
    parser.add_argument('--model_name', type=str, default=DEFAULT_MODEL_NAME, help='YOLO model name or path to weights.')
    parser.add_argument('--conf_thresh', type=float, default=CONF_THRESHOLD, help='Confidence threshold for object detection.')
    parser.add_argument('--iou_thresh', type=float, default=IOU_THRESHOLD, help='IoU threshold for matching objects.')
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of the GPU to use. Set to None to use CPU.')
    parser.add_argument('--save_examples', type=int, default=10, help='Number of visual examples to save. Set to 0 to disable.')
    
    # --- New arguments for realistic evaluation ---
    parser.add_argument('--patch_scale', type=float, default=0.4, help='Ratio of patch area to target object area (e.g., 0.4 means patch area is 40%% of target area).')
    parser.add_argument('--occlusion_level', type=float, default=0.4, help='How much the patch should overlap the target. 0.0=touching edge, 1.0=centered on edge.')

    args = parser.parse_args()
    
    try:
        evaluate_patch(args)
    except Exception as e:
        console.print(f"\n💥 [bold red]An unexpected error occurred during evaluation![/bold red]")
        console.print_exception(show_locals=True)
