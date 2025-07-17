import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import os
import random
import numpy as np
from tqdm import tqdm
from datetime import datetime
import sys

# --- This file contains the shared, definitive evaluation logic ---

class VisDroneValidationDataset(Dataset):
    """Loads images and their corresponding ground-truth annotations from the VisDrone validation set."""
    def __init__(self, root_dir, annotation_dir='annotations_v11'):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images')
        self.annotation_dir = os.path.join(root_dir, annotation_dir)
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith(('.jpg', '.png'))])
        
        if not os.path.isdir(self.annotation_dir):
            print(f"⚠️ Warning: Annotation directory not found at '{self.annotation_dir}'.")
            self.label_dir = None
        else:
            self.label_dir = self.annotation_dir

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        
        gt_boxes = []
        if self.label_dir:
            label_filename = os.path.splitext(self.image_files[idx])[0] + '.txt'
            label_path = os.path.join(self.label_dir, label_filename)
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        try:
                            parts = line.strip().split(',')
                            if len(parts) >= 4:
                                x1, y1, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                                gt_boxes.append([x1, y1, x1 + w, y1 + h])
                        except (ValueError, IndexError):
                            continue
        return image, gt_boxes, img_path

def apply_patch_to_image_partial(image_pil, patch_pil, box):
    """
    Applies a patch to partially cover a specified bounding box, keeping the patch square
    and placing it near the center of the object.
    """
    img_w, img_h = image_pil.size
    patched_image = image_pil.copy()
    x1, y1, x2, y2 = map(int, box)
    box_w, box_h = x2 - x1, y2 - y1

    if box_w <= 0 or box_h <= 0: return patched_image
    
    scale = random.uniform(0.4, 0.7)
    
    patch_side_length = int(min(box_w, box_h) * scale)
    if patch_side_length == 0: return patched_image
    
    resized_patch = patch_pil.resize((patch_side_length, patch_side_length), Image.LANCZOS)
    
    center_x = x1 + box_w / 2
    center_y = y1 + box_h / 2

    base_paste_x = center_x - (patch_side_length / 2)
    base_paste_y = center_y - (patch_side_length / 2)

    max_jitter = int(patch_side_length * 0.2)
    jitter_x = random.randint(-max_jitter, max_jitter)
    jitter_y = random.randint(-max_jitter, max_jitter)

    paste_x = int(base_paste_x + jitter_x)
    paste_y = int(base_paste_y + jitter_y)
    
    paste_x = max(0, min(paste_x, img_w - patch_side_length))
    paste_y = max(0, min(paste_y, img_h - patch_side_length))
    
    patched_image.paste(resized_patch, (paste_x, paste_y), resized_patch if resized_patch.mode == 'RGBA' else None)
    
    return patched_image

def draw_boxes(image_pil, boxes, color="lime", confidences=None):
    """Draws bounding boxes and optional confidences on a PIL image."""
    draw = ImageDraw.Draw(image_pil)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        if confidences is not None:
            label = f"Conf: {confidences[i]:.2f}"
            try:
                text_bbox = draw.textbbox((x1, y1 - 12), label)
                draw.rectangle(text_bbox, fill=color)
                draw.text((x1, y1 - 12), label, fill="black")
            except Exception:
                draw.text((x1, y1 - 5), label, fill=color)

def run_evaluation(model, patch_path, dataset_path, conf_threshold, num_eval_images, seed, 
                   status_queue=None, gpu_id=None, visualize=False, visual_limit=10):
    """
    The single, definitive evaluation function, optimized for speed and live progress reporting.
    """
    if seed is not None:
        print(f"Running evaluation with fixed seed: {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    try:
        patch_pil = Image.open(patch_path).convert("RGB")
    except FileNotFoundError:
        print(f"❌ Evaluation Error: Patch not found at '{patch_path}'.")
        return 0.0

    dataset = VisDroneValidationDataset(root_dir=dataset_path)
    if len(dataset) == 0:
        print(f"❌ Error: No validation images found in '{dataset_path}'.")
        return 0.0

    output_dir = None
    if visualize:
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        patch_name = os.path.splitext(os.path.basename(patch_path))[0]
        output_dir = os.path.join("evaluation_visuals", f"{timestamp}_{patch_name}")
        os.makedirs(output_dir, exist_ok=True)
        print(f"💾 Saving visualization images to '{output_dir}'")

    eval_indices = range(len(dataset))
    print(f"  - Evaluating on all {len(dataset)} validation images.")

    total_attacks, successful_attacks, saved_visuals_count = 0, 0, 0
    
    for idx, _ in enumerate(tqdm(eval_indices, desc="  Evaluating Patch", leave=False, disable=True)):
        original_image_pil, gt_boxes, img_path = dataset[idx]
        
        if status_queue and (idx % 5 == 0 or idx == len(eval_indices) - 1):
            progress_percent = (idx + 1) / len(eval_indices) * 100
            status_queue.put({
                'gpu_id': gpu_id,
                'status': 'Evaluating',
                'progress': f'{progress_percent:.1f}%'
            })

        if not gt_boxes:
            continue
        
        results_before_list = model(original_image_pil, conf=conf_threshold, verbose=False)
        num_boxes_before = len(results_before_list[0].boxes)

        if num_boxes_before == 0:
            continue

        patched_images_batch = [apply_patch_to_image_partial(original_image_pil, patch_pil, gt_box) for gt_box in gt_boxes]

        if not patched_images_batch:
            continue

        total_attacks += len(patched_images_batch)

        results_after_list = model(patched_images_batch, conf=conf_threshold, verbose=False)

        for i, results_after in enumerate(results_after_list):
            num_boxes_after = len(results_after.boxes)
            is_success = num_boxes_after < num_boxes_before
            
            if is_success:
                successful_attacks += 1
            
            if visualize and (visual_limit == -1 or saved_visuals_count < visual_limit):
                attack_status = "SUCCESS" if is_success else "FAIL"
                gt_box_for_vis = gt_boxes[i]
                
                img_before_with_box = original_image_pil.copy()
                draw_boxes(img_before_with_box, [gt_box_for_vis], color="red")

                img_after_with_detections = Image.fromarray(results_after.orig_img[:,:,::-1])
                draw_boxes(img_after_with_detections, 
                           results_after.boxes.xyxy.cpu().numpy(), 
                           color="lime", 
                           confidences=results_after.boxes.conf.cpu().numpy())
                
                comparison_img = Image.new('RGB', (original_image_pil.width * 2, original_image_pil.height))
                comparison_img.paste(img_before_with_box, (0, 0))
                comparison_img.paste(img_after_with_detections, (original_image_pil.width, 0))
                
                draw = ImageDraw.Draw(comparison_img)
                draw.text((10, 10), "Before (Ground Truth)", fill="red")
                draw.text((original_image_pil.width + 10, 10), f"After (YOLO Detections) - {attack_status}", fill="lime")

                base_filename = os.path.splitext(os.path.basename(img_path))[0]
                comparison_img.save(os.path.join(output_dir, f"{base_filename}_attack_{i}_{attack_status}.png"))
                saved_visuals_count += 1

    if total_attacks == 0: return 0.0
    return (successful_attacks / total_attacks) * 100.0
