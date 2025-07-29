import torch
import torchvision.transforms as T
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
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn, SpinnerColumn
from rich.panel import Panel
from rich.live import Live

# --- Configuration ---
DEFAULT_MODEL_NAME = 'yolov11n.pt'
DEFAULT_DATASET_PATH = 'VisDrone2019-DET-val'
DEFAULT_PATCH_PATH = 'runs/best_patch.png'

# --- Evaluation Parameters ---
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5
MIN_TARGET_PIXELS = 64 * 64 # Ignore very small objects for targeting

# --- Initialize Rich Console ---
console = Console()

def calculate_iou(box1, box2):
    """Calculates Intersection over Union (IoU) for two bounding boxes."""
    x1_inter, y1_inter = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2_inter, y2_inter = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter_area / (box1_area + box2_area - inter_area + 1e-6)

class EvaluationDataset(Dataset):
    """Dataset class for loading evaluation images."""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_dir = os.path.join(root_dir, 'images')
        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"Image directory not found at: {self.image_dir}")
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        console.print(f"✅ [green]Evaluation dataset initialized. Found {len(self.image_files)} images.[/green]")

    def __len__(self): return len(self.image_files)
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            console.print(f"⚠️ [yellow]Could not load image {img_path}. Skipping. Error: {e}[/yellow]")
            return None, None
        return self.transform(image) if self.transform else image, img_name

def collate_fn_eval(batch):
    """Custom collate function to filter out None values from failed image loads."""
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch: return None, None
    images, names = zip(*batch)
    return torch.stack(images, 0), names

def run_evaluation_for_scale(model, dataloader, base_patch_image, patch_scale, device, conf_thresh, progress, task_id):
    """
    Runs a full evaluation pass for a single, specific patch scale.
    
    Args:
        model: The loaded YOLO model.
        dataloader: The DataLoader for the evaluation dataset.
        base_patch_image (PIL.Image): The adversarial patch to test.
        patch_scale (float): The ratio of patch area to target area for this run.
        device: The torch device to run on.
        conf_thresh (float): Confidence threshold for detections.
        progress: The Rich Progress instance for updating the UI.
        task_id: The ID of the progress bar task for this scale.

    Returns:
        A dictionary containing the evaluation metrics for this scale.
    """
    total_targets_attacked = 0
    successful_attacks = 0
    total_scene_confidence_before = 0
    total_scene_confidence_after = 0
    total_false_positives = 0

    for image_tensor, img_names in dataloader:
        if image_tensor is None:
            progress.update(task_id, advance=1)
            continue
        
        image_tensor = image_tensor.to(device)

        # 1. Analyze Clean Scene
        with torch.no_grad():
            results_clean = model(image_tensor.clamp(0, 1), verbose=False, conf=conf_thresh)
        
        clean_boxes = results_clean[0].boxes.xyxy
        clean_confs = results_clean[0].boxes.conf
        total_scene_confidence_before += clean_confs.sum().item()

        valid_targets_mask = (clean_boxes[:, 2] - clean_boxes[:, 0]) * (clean_boxes[:, 3] - clean_boxes[:, 1]) > MIN_TARGET_PIXELS
        if not valid_targets_mask.any():
            progress.update(task_id, advance=1)
            continue

        valid_target_indices = torch.where(valid_targets_mask)[0]
        target_idx = valid_target_indices[random.randint(0, len(valid_target_indices) - 1)]
        target_box = clean_boxes[target_idx]
        total_targets_attacked += 1

        # 2. Prepare and Place Scaled Patch
        target_w, target_h = target_box[2] - target_box[0], target_box[3] - target_box[1]
        patch_base_size = int(math.sqrt(max(1.0, (target_w * target_h) * patch_scale)))
        patch_base_size = max(patch_base_size, 10) # Ensure patch is not invisibly small

        patch_transform = T.Compose([
            T.Resize((patch_base_size, patch_base_size), antialias=True),
            T.ToTensor()
        ])
        patch_tensor = patch_transform(base_patch_image).to(device)
        
        patched_image = image_tensor.clone().squeeze(0)
        
        center_x = (target_box[0] + target_box[2]) / 2
        center_y = (target_box[1] + target_box[3]) / 2
        
        px = int(center_x - patch_base_size / 2)
        py = int(center_y - patch_base_size / 2)
        
        x1, y1 = max(0, px), max(0, py)
        x2, y2 = min(640, px + patch_base_size), min(640, py + patch_base_size)
        crop_x1, crop_y1 = max(0, x1 - px), max(0, y1 - py)
        crop_x2, crop_y2 = patch_base_size - max(0, (px + patch_base_size) - 640), patch_base_size - max(0, (py + patch_base_size) - 640)
        
        if (y2 > y1) and (x2 > x1):
            patch_slice = patch_tensor[:, crop_y1:crop_y2, crop_x1:crop_x2]
            if patch_slice.shape[1] == (y2-y1) and patch_slice.shape[2] == (x2-x1):
                patched_image[:, y1:y2, x1:x2] = patch_slice

        # 3. Analyze Patched Scene
        with torch.no_grad():
            results_patched = model(patched_image.unsqueeze(0).clamp(0, 1), verbose=False, conf=conf_thresh)
        
        patched_boxes, patched_confs = results_patched[0].boxes.xyxy, results_patched[0].boxes.conf
        total_scene_confidence_after += patched_confs.sum().item()

        # 4. Evaluate Attack Success and Side Effects
        best_iou = 0
        for p_box in patched_boxes:
            iou = calculate_iou(target_box.cpu().numpy(), p_box.cpu().numpy())
            if iou > best_iou: best_iou = iou
        
        if best_iou < IOU_THRESHOLD:
            successful_attacks += 1

        for p_box in patched_boxes:
            is_fp = True
            for c_box in clean_boxes:
                if calculate_iou(p_box.cpu().numpy(), c_box.cpu().numpy()) > IOU_THRESHOLD:
                    is_fp = False
                    break
            if is_fp:
                total_false_positives += 1
        
        progress.update(task_id, advance=1)

    # Calculate final metrics for this scale
    asr = (successful_attacks / total_targets_attacked) * 100 if total_targets_attacked > 0 else 0
    disruption = (1 - (total_scene_confidence_after / total_scene_confidence_before)) * 100 if total_scene_confidence_before > 0 else 0
    
    return {
        "patch_scale": patch_scale,
        "asr": asr,
        "disruption": disruption,
        "false_positives": total_false_positives,
        "targets_attacked": total_targets_attacked
    }

def optimize_patch_size(args):
    """Main function to iterate through patch scales and find the optimum."""
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    console.print(Panel(f"🚀 [bold magenta]Starting Patch Size Optimization[/bold magenta]\n"
                        f"   - [b]Device[/b]: [cyan]{device.type.upper()}[/cyan]\n"
                        f"   - [b]Patch[/b]: [cyan]{args.patch_path}[/cyan]\n"
                        f"   - [b]Model[/b]: [cyan]{args.model_name}[/cyan]\n"
                        f"   - [b]Scale Range[/b]: [cyan]{args.min_scale:.2f} to {args.max_scale:.2f}[/cyan]\n"
                        f"   - [b]Scale Step[/b]: [cyan]{args.scale_step:.2f}[/cyan]",
                        title="[yellow]Optimization Configuration[/yellow]", border_style="yellow"))

    model = YOLO(args.model_name).to(device)
    model.model.eval()

    base_patch_image = Image.open(args.patch_path).convert("RGB")
    dataset = EvaluationDataset(root_dir=args.dataset_path, transform=T.Compose([T.Resize((640, 640)), T.ToTensor()]))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn_eval)
    
    scales_to_test = np.arange(args.min_scale, args.max_scale + args.scale_step, args.scale_step)
    all_results = []

    # Setup Rich Live display
    overall_progress = Progress(TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(), TimeRemainingColumn())
    evaluation_progress = Progress(TextColumn("    {task.description}"), BarColumn(), MofNCompleteColumn())
    live_layout = Table.grid(expand=True)
    live_layout.add_row(Panel(overall_progress, title="[b]Overall Progress[/b]"))
    live_layout.add_row(Panel(evaluation_progress, title="[b]Current Scale Evaluation[/b]"))

    with Live(live_layout, console=console, screen=False, vertical_overflow="visible"):
        overall_task = overall_progress.add_task("[green]Testing Scales...", total=len(scales_to_test))

        for scale in scales_to_test:
            eval_task_desc = f"[cyan]Evaluating Scale: {scale:.2f}"
            eval_task = evaluation_progress.add_task(eval_task_desc, total=len(dataloader))
            
            # Run the evaluation for the current scale
            result = run_evaluation_for_scale(model, dataloader, base_patch_image, scale, device, args.conf_thresh, evaluation_progress, eval_task)
            
            all_results.append(result)
            evaluation_progress.remove_task(eval_task)
            overall_progress.update(overall_task, advance=1)

    # --- Final Report ---
    console.print("\n" + "="*80)
    console.print("📊 [bold green]Optimization Complete: Final Results[/bold green]")
    console.print("="*80)

    results_table = Table(title="Patch Scale Performance Comparison", show_header=True, header_style="bold magenta")
    results_table.add_column("Patch Scale", style="cyan", justify="center")
    results_table.add_column("Attack Success Rate (ASR %)", style="green", justify="center")
    results_table.add_column("Scene Disruption (%)", style="yellow", justify="center")
    results_table.add_column("False Positives", style="red", justify="center")
    results_table.add_column("Targets Attacked", style="blue", justify="center")
    
    best_asr = -1.0
    best_scale_for_asr = None

    # Sort results by ASR for clearer presentation
    all_results.sort(key=lambda x: x['asr'], reverse=True)

    for res in all_results:
        is_best = ""
        if res['asr'] > best_asr:
            best_asr = res['asr']
            best_scale_for_asr = res['patch_scale']
            is_best = "⭐ "

        results_table.add_row(
            f"{is_best}{res['patch_scale']:.2f}",
            f"{res['asr']:.2f}%",
            f"{res['disruption']:.2f}%",
            f"{res['false_positives']}",
            f"{res['targets_attacked']}"
        )

    console.print(results_table)
    
    if best_scale_for_asr is not None:
        summary_panel = Panel(
            f"The highest Attack Success Rate (ASR) of [bold green]{best_asr:.2f}%[/bold green] was achieved with a patch scale of [bold cyan]{best_scale_for_asr:.2f}[/bold cyan].\n\n"
            f"[italic]This scale represents the ratio of the patch's area relative to the target object's area.[/italic]",
            title="[bold blue]Optimal Scale Recommendation[/bold blue]",
            border_style="blue"
        )
        console.print(summary_panel)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Find the optimal adversarial patch size by testing a range of scales.")
    # --- Core Arguments ---
    parser.add_argument('--patch_path', type=str, default=DEFAULT_PATCH_PATH, help='Path to the adversarial patch image file.')
    parser.add_argument('--dataset_path', type=str, default=DEFAULT_DATASET_PATH, help='Path to the root of the evaluation dataset.')
    parser.add_argument('--model_name', type=str, default=DEFAULT_MODEL_NAME, help='YOLO model name or path to weights.')
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of the GPU to use.')
    
    # --- Optimization Arguments ---
    parser.add_argument('--min_scale', type=float, default=0.2, help='Minimum patch-to-target area ratio to test.')
    parser.add_argument('--max_scale', type=float, default=1.0, help='Maximum patch-to-target area ratio to test.')
    parser.add_argument('--scale_step', type=float, default=0.1, help='Step size to increment the scale ratio.')

    # --- Evaluation Arguments ---
    parser.add_argument('--conf_thresh', type=float, default=CONF_THRESHOLD, help='Confidence threshold for object detection.')
    
    args = parser.parse_args()
    
    try:
        optimize_patch_size(args)
    except Exception as e:
        console.print(f"\n💥 [bold red]An unexpected error occurred during optimization![/bold red]")
        console.print_exception(show_locals=False)
