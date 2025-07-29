import torch
import torchvision.transforms.functional as TF
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import os
import argparse
import random
import math
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
import numpy as np

# --- Initialize Rich Console for clean terminal output ---
console = Console()

def load_model(model_path, device):
    """Loads the YOLO model from the specified path."""
    console.print(f"🧠 [bold cyan]Loading model from: [green]{model_path}[/green]...[/bold cyan]")
    try:
        model = YOLO(model_path)
        model.to(device)
        console.print("✅ [bold green]Model loaded successfully.[/bold green]")
        return model
    except Exception as e:
        console.print(f"💥 [bold red]Error loading model: {e}[/bold red]")
        exit()

def load_patch(patch_path, device):
    """Loads the adversarial patch image and converts it to a tensor."""
    console.print(f"🎨 [bold cyan]Loading patch from: [green]{patch_path}[/green]...[/bold cyan]")
    try:
        patch_image = Image.open(patch_path).convert("RGB")
        patch_tensor = TF.to_tensor(patch_image).to(device)
        console.print("✅ [bold green]Patch loaded successfully.[/bold green]")
        return patch_tensor
    except Exception as e:
        console.print(f"💥 [bold red]Error loading patch: {e}[/bold red]")
        exit()

def apply_patch_to_object(image_tensor, patch_tensor, box, coverage):
    """
    Applies the patch to a specific bounding box on the image tensor,
    scaling the patch based on the desired coverage of the object's area.
    """
    patched_image = image_tensor.clone()
    x1, y1, x2, y2 = map(int, box)
    
    target_w, target_h = x2 - x1, y2 - y1
    if target_w <= 0 or target_h <= 0:
        return patched_image

    # Calculate patch size based on the percentage of the object's AREA
    patch_area = target_w * target_h * (coverage / 100.0)
    
    # Maintain the aspect ratio of the original patch
    patch_aspect_ratio = patch_tensor.shape[2] / patch_tensor.shape[1]  # width / height
    patch_h_new = int(math.sqrt(patch_area / patch_aspect_ratio))
    patch_w_new = int(patch_aspect_ratio * patch_h_new)

    if patch_w_new == 0 or patch_h_new == 0:
        return patched_image
        
    # Resize the patch to the calculated dimensions
    resized_patch = TF.resize(patch_tensor.unsqueeze(0), (patch_h_new, patch_w_new), antialias=True).squeeze(0)

    # Place the patch at the center of the target object
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    patch_x = max(0, min(center_x - patch_w_new // 2, image_tensor.shape[2] - patch_w_new))
    patch_y = max(0, min(center_y - patch_h_new // 2, image_tensor.shape[1] - patch_h_new))

    # Apply the patch to the image tensor
    patched_image[:, patch_y:patch_y + patch_h_new, patch_x:patch_x + patch_w_new] = resized_patch
    
    return patched_image

def draw_detections(image, detections, class_names):
    """Draws bounding boxes and class labels on a PIL image."""
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    for box, conf, cls_id in zip(detections.xyxy, detections.conf, detections.cls):
        x1, y1, x2, y2 = map(int, box)
        label = f"{class_names[int(cls_id)]} {conf:.2f}"
        
        # Use a random color for each class for better visualization
        color = tuple(np.random.randint(100, 255, 3))
        
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw a filled rectangle as a background for the text
        text_size = draw.textbbox((0, 0), label, font=font)
        text_w, text_h = text_size[2] - text_size[0], text_size[3] - text_size[1]
        draw.rectangle([x1, y1 - text_h - 5, x1 + text_w, y1], fill=color)
        draw.text((x1, y1 - text_h - 5), label, fill="white", font=font)
        
    return image

def create_comparison_image(before_img, after_img):
    """Creates a side-by-side comparison image of detections before and after the patch."""
    dst = Image.new('RGB', (before_img.width + after_img.width, before_img.height))
    dst.paste(before_img, (0, 0))
    dst.paste(after_img, (before_img.width, 0))
    return dst

def iou(boxA, boxB):
    """Calculates the Intersection over Union (IoU) of two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    iou_val = interArea / float(boxAArea + boxBArea - interArea + 1e-6) # Add epsilon to avoid division by zero
    return iou_val

def evaluate(model, patch_tensor, image_paths, output_dir, target_classes, coverage, conf_thresh, iou_thresh):
    """The main evaluation loop to test the patch's effectiveness."""
    device = patch_tensor.device
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store the results of the evaluation
    results_summary = {
        "total_targeted": 0,
        "hidden": 0,          # Object completely disappears
        "misclassified": 0,   # Object detected but with the wrong class
        "disrupted": 0,       # Object detected correctly but with significantly lower confidence
        "failed": 0           # Object detected correctly with high confidence
    }

    for img_path in track(image_paths, description="[cyan]Evaluating Images...[/cyan]"):
        try:
            original_pil_image = Image.open(img_path).convert("RGB")
        except Exception as e:
            console.log(f"⚠️ [yellow]Could not open image {img_path}. Skipping. Error: {e}[/yellow]")
            continue
            
        original_tensor = TF.to_tensor(original_pil_image).to(device)

        # 1. Get baseline detections on the original image
        baseline_results = model(original_pil_image, conf=conf_thresh, iou=iou_thresh, verbose=False)[0]
        
        # 2. Identify target objects to attack
        target_boxes = []
        if not target_classes:  # Target all detected objects if no specific classes are provided
            target_boxes = [(box, cls) for box, cls in zip(baseline_results.boxes.xyxy, baseline_results.boxes.cls)]
        else:
            for box, cls in zip(baseline_results.boxes.xyxy, baseline_results.boxes.cls):
                if model.names[int(cls)] in target_classes:
                    target_boxes.append((box, cls))
        
        if not target_boxes:
            continue
            
        results_summary["total_targeted"] += len(target_boxes)
        
        # Apply the patch to all identified target objects
        patched_tensor = original_tensor.clone()
        for box, _ in target_boxes:
            patched_tensor = apply_patch_to_object(patched_tensor, patch_tensor, box, coverage)
            
        patched_pil_image = TF.to_pil_image(patched_tensor.cpu())

        # 3. Get detections on the patched image
        patched_results = model(patched_pil_image, conf=conf_thresh, iou=iou_thresh, verbose=False)[0]
        
        # 4. Analyze the outcome for each targeted object
        for original_box, original_cls_id in target_boxes:
            found_match = False
            for patched_box, patched_conf, patched_cls_id in zip(patched_results.boxes.xyxy, patched_results.boxes.conf, patched_results.boxes.cls):
                if iou(original_box.cpu().numpy(), patched_box.cpu().numpy()) > iou_thresh:
                    found_match = True
                    if int(patched_cls_id) != int(original_cls_id):
                        results_summary["misclassified"] += 1
                    else:
                        # Find original confidence to check for disruption
                        original_conf = next((c for b, c in zip(baseline_results.boxes.xyxy, baseline_results.boxes.conf) if iou(original_box.cpu().numpy(), b.cpu().numpy()) > 0.99), 0)
                        if patched_conf < original_conf * 0.5:
                             results_summary["disrupted"] += 1
                        else:
                             results_summary["failed"] += 1
                    break
            
            if not found_match:
                results_summary["hidden"] += 1
        
        # 5. Save a side-by-side comparison image
        img_before = draw_detections(original_pil_image, baseline_results.boxes, model.names)
        img_after = draw_detections(patched_pil_image, patched_results.boxes, model.names)
        comparison_img = create_comparison_image(img_before, img_after)
        base_filename = os.path.splitext(os.path.basename(img_path))[0]
        comparison_img.save(os.path.join(output_dir, f"{base_filename}_comparison.jpg"))
        
    return results_summary

def main():
    parser = argparse.ArgumentParser(description="Evaluate an adversarial patch against a YOLO object detection model.")
    parser.add_argument("--model", type=str, required=True, help="Path to the YOLO model file (e.g., yolov8n.pt).")
    parser.add_argument("--patch", type=str, required=True, help="Path to the adversarial patch image file.")
    parser.add_argument("--output", type=str, default="evaluation_results", help="Directory to save comparison images and results.")
    parser.add_argument("--target_classes", nargs='+', default=[], help="A list of class names to target. If empty, all objects are targeted.")
    parser.add_argument("--coverage", type=float, default=35.0, help="Percentage of the object's bounding box AREA to cover with the patch.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for object detection.")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for matching detections.")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # NOTE: The image directory is hardcoded for simplicity.
    # For a more flexible script, this could be made a command-line argument.
    image_dir = "VisDrone2019-DET-val/images/"

    console.print(Panel(f"🚀 [bold]Starting Adversarial Patch Evaluation[/bold] 🚀\n"
                        f"  - [b]Device[/b]: [cyan]{device.type.upper()}[/cyan]\n"
                        f"  - [b]Image Source[/b]: [cyan]{image_dir}[/cyan]\n"
                        f"  - [b]Target Classes[/b]: [cyan]{args.target_classes or 'All'}[/cyan]\n"
                        f"  - [b]Patch Coverage[/b]: [cyan]{args.coverage}%[/cyan]",
                        title="[yellow]Evaluation Configuration[/yellow]", border_style="yellow"))

    model = load_model(args.model, device)
    patch = load_patch(args.patch, device)
    
    # Get all image paths from the directory
    if not os.path.isdir(image_dir):
        console.print(f"[bold red]Error: Image directory not found at '{image_dir}'[/bold red]")
        return
    all_image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not all_image_files:
        console.print(f"[bold red]No images found in '{image_dir}'[/bold red]")
        return
    console.print(f"✅ [green]Found {len(all_image_files)} images to evaluate.[/green]")

    summary = evaluate(model, patch, all_image_files, args.output, args.target_classes, args.coverage, args.conf, args.iou)

    # Print the final summary table
    if summary["total_targeted"] > 0:
        table = Table(title="[bold blue]Evaluation Summary[/bold blue]", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="dim", width=25)
        table.add_column("Count", justify="right")
        table.add_column("Percentage", justify="right")
        
        total = summary['total_targeted']
        
        table.add_row("Total Objects Targeted", str(total), "100.00%")
        table.add_row("[green]Success (Hidden)[/green]", str(summary['hidden']), f"{(summary['hidden']/total)*100:.2f}%")
        table.add_row("[yellow]Success (Misclassified)[/yellow]", str(summary['misclassified']), f"{(summary['misclassified']/total)*100:.2f}%")
        table.add_row("[blue]Success (Disrupted)[/blue]", str(summary['disrupted']), f"{(summary['disrupted']/total)*100:.2f}%")
        table.add_row(end_section=True)
        
        total_success_count = summary['hidden'] + summary['misclassified'] + summary['disrupted']
        table.add_row("[bold green]Total Success Rate[/bold green]", str(total_success_count), f"{(total_success_count / total) * 100:.2f}%")
        table.add_row("[bold red]Failed (Detected) Rate[/bold red]", str(summary['failed']), f"{(summary['failed']/total)*100:.2f}%")
        
        console.print(table)
        console.print(f"\n💾 [bold]Comparison images saved in: [green]{os.path.abspath(args.output)}[/green][/bold]")
    else:
        console.print("[bold yellow]Evaluation finished, but no target objects were found to attack in the selected images.[/bold yellow]")

if __name__ == '__main__':
    main()
