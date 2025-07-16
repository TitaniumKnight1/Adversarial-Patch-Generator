import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from ultralytics import YOLO
from PIL import Image
import os
import random
from tqdm import tqdm
import argparse
import torch.backends.cudnn as cudnn
from tensorboard import program
import requests
import traceback
import sys
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime
# Import the evaluation logic from the user-provided file
from evaluation_logic import run_evaluation

# --- Configuration (Defaults) ---
DATASET_PATH = 'VisDrone2019-DET-train'
VALIDATION_DATASET_PATH = 'VisDrone2019-DET-val'
MODEL_NAME = 'yolov11n.pt'
# The resolution at which the patch is trained. It will be resized on-the-fly.
PATCH_RESOLUTION = 100

# --- Base values for dynamic learning rate scaling ---
BASE_LR = 0.0001
BASE_BATCH_SIZE = 8.0

# --- Smart Training Config ---
PLATEAU_PATIENCE = 5
CHECKPOINT_FILE = "patch_checkpoint.pth"

# --- Early Stopping & Evaluation Configuration ---
EVAL_PATIENCE = 10 
# ✅ NEW: Confidence threshold for the evaluation model
EVAL_CONF_THRESHOLD = 0.25

# --- Dataset Classes ---
class VisDroneDataset(Dataset):
    """Loads images AND their corresponding bounding box annotations."""
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
        image = Image.open(img_path).convert("RGB")
        original_size = image.size
        boxes = []
        annotation_name = os.path.splitext(img_name)[0] + '.txt'
        annotation_path = os.path.join(self.annotation_dir, annotation_name)
        if os.path.exists(annotation_path):
            with open(annotation_path, 'r') as f:
                for line in f.readlines():
                    try:
                        parts = line.strip().split(',')
                        x1, y1, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                        boxes.append([x1, y1, x1 + w, y1 + h])
                    except (ValueError, IndexError):
                        continue
        boxes = torch.tensor(boxes, dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        if boxes.nelement() > 0:
            scale_x, scale_y = 640 / original_size[0], 640 / original_size[1]
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
        return image, boxes

class DummyDataset(Dataset):
    """A simple dummy dataset for the autotune memory test."""
    def __init__(self, length=256): self.length = length
    def __len__(self): return self.length
    def __getitem__(self, idx): return torch.rand(3, 640, 640)

# --- Autotune Function ---
def autotune_batch_size(device, model, dataset, initial_batch_size=2):
    batch_size = initial_batch_size
    print(f"🚀 Starting batch size autotune from size {batch_size}...")
    while True:
        try:
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            images = next(iter(dataloader)).to(device)
            dummy_patch = torch.rand((3, PATCH_RESOLUTION, PATCH_RESOLUTION), device=device, requires_grad=True)
            optimizer = torch.optim.Adam([dummy_patch], lr=0.01)
            for j in range(min(images.size(0), batch_size)):
                 patch_side_length = random.randint(50, 150)
                 resized_patch = TF.resize(dummy_patch, [patch_side_length, patch_side_length], antialias=True)
                 x, y = random.randint(0, 640 - patch_side_length), random.randint(0, 640 - patch_side_length)
                 images[j, :, y:y+patch_side_length, x:x+patch_side_length] = resized_patch
            with torch.amp.autocast(device.type):
                raw_preds = model.model(images)[0].transpose(1, 2)
                loss = torch.mean(torch.max(raw_preds[..., 4:], dim=-1)[0])
            loss.backward()
            optimizer.zero_grad(set_to_none=True)
            print(f"✅ Batch size {batch_size} fits in memory. Trying next size...")
            del images, raw_preds, loss, dummy_patch, optimizer
            batch_size *= 2
        except torch.cuda.OutOfMemoryError:
            max_size = batch_size // 2; print(f"⚠️ OOM at {batch_size}. Optimal: {max_size}."); torch.cuda.empty_cache(); return max_size
        except StopIteration:
            print(f"✅ Batch size {batch_size} fits, but dataset is too small to double."); return batch_size

# --- Dummy Writer for Headless Mode ---
class DummyWriter:
    """A no-op SummaryWriter for when TensorBoard is disabled."""
    def add_scalar(self, *args, **kwargs): pass
    def add_image(self, *args, **kwargs): pass
    def close(self): pass

# --- Adversarial Patch Training ---
def train_adversarial_patch(batch_size, learning_rate, log_dir, device, num_eval_images, use_tensorboard, resume_path=None, starter_image_path=None):
    if device.type == 'cuda': cudnn.benchmark = True
    
    writer = SummaryWriter(log_dir=log_dir) if use_tensorboard else DummyWriter()
    
    training_model = YOLO(MODEL_NAME)
    training_model.to(device)
    training_model.model.train()

    if device.type == 'cuda':
        try:
            training_model.model = torch.compile(training_model.model)
            print("✅ Model compiled successfully with torch.compile().")
        except Exception as e:
            print(f"⚠️ torch.compile() failed: {e}. Running without compilation.")

    if starter_image_path and os.path.exists(starter_image_path):
        print(f"Initializing patch from starter image: {starter_image_path}")
        starter_image = Image.open(starter_image_path).convert("RGB")
        transform_starter = T.Compose([T.Resize((PATCH_RESOLUTION, PATCH_RESOLUTION)), T.ToTensor()])
        adversarial_patch = transform_starter(starter_image).to(device)
        adversarial_patch.requires_grad_(True)
    else:
        print("Initializing patch with random noise.")
        adversarial_patch = torch.rand((3, PATCH_RESOLUTION, PATCH_RESOLUTION), device=device, requires_grad=True)

    print(f"Initializing optimizer with learning rate: {learning_rate:.6f}")
    optimizer = torch.optim.Adam([adversarial_patch], lr=learning_rate)
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == 'cuda'))
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=PLATEAU_PATIENCE)

    epoch = 0
    best_success_rate = -1.0
    epochs_no_improve = 0
    
    if resume_path and os.path.exists(resume_path):
        print(f"Resuming training from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        adversarial_patch.data = checkpoint['patch_state_dict'].to(device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        best_success_rate = checkpoint.get('best_success_rate', -1.0)
        epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
        print(f"Resumed from epoch {epoch} with best success rate of {best_success_rate:.2f}%.")

    transform = T.Compose([T.Resize((640, 640)), T.ToTensor()])
    dataset = VisDroneDataset(root_dir=DATASET_PATH, transform=transform)
    num_workers = os.cpu_count() // 2 if os.cpu_count() else 4
    pin_memory = (device.type == 'cuda')
    def collate_fn(batch):
        images, boxes = [item[0] for item in batch], [item[1] for item in batch]
        return torch.stack(images, 0), boxes
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn)

    print(f"\nStarting adversarial patch training with batch size: {batch_size}")
    print(f"Using device: {device.type.upper()}. Using {num_workers} data loader workers.")

    while True:
        epoch += 1
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} (dynamic)")

        for i, (images, gt_boxes_batch) in enumerate(progress_bar):
            images = images.to(device, non_blocking=pin_memory)
            
            for img_idx in range(images.size(0)):
                gt_boxes = gt_boxes_batch[img_idx]
                if len(gt_boxes) > 0:
                    box_idx = random.randint(0, len(gt_boxes) - 1)
                    box = gt_boxes[box_idx]
                    x1, y1, x2, y2 = box
                    box_w, box_h = x2 - x1, y2 - y1
                    if box_w <= 0 or box_h <= 0: continue
                    base_size = min(box_w.item(), box_h.item())
                    scale = random.uniform(0.4, 0.7)
                    patch_side_length = int(base_size * scale)
                    if patch_side_length == 0: continue
                    resized_patch = TF.resize(adversarial_patch, [patch_side_length, patch_side_length], antialias=True)
                    center_x, center_y = x1 + box_w / 2, y1 + box_h / 2
                    base_paste_x, base_paste_y = center_x - (patch_side_length / 2), center_y - (patch_side_length / 2)
                    max_jitter = int(patch_side_length * 0.2)
                    jitter_x, jitter_y = random.randint(-max_jitter, max_jitter), random.randint(-max_jitter, max_jitter)
                    paste_x, paste_y = int(base_paste_x + jitter_x), int(base_paste_y + jitter_y)
                    paste_x = max(0, min(paste_x, images.shape[3] - patch_side_length))
                    paste_y = max(0, min(paste_y, images.shape[2] - patch_side_length))
                    images[img_idx, :, paste_y:paste_y+patch_side_length, paste_x:paste_x+patch_side_length] = resized_patch
                else:
                    x_start, y_start = random.randint(0, 640 - 100), random.randint(0, 640 - 100)
                    images[img_idx, :, y_start:y_start+100, x_start:x_start+100] = TF.resize(adversarial_patch, [100, 100], antialias=True)

            with torch.amp.autocast(device.type):
                raw_preds = training_model.model(images)[0].transpose(1, 2)
                loss = torch.mean(torch.max(raw_preds[..., 4:], dim=-1)[0])
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(set_to_none=True)
            adversarial_patch.data.clamp_(0, 1)
            total_loss += loss.item()
            current_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix(adv_loss=f"{total_loss/(i+1):.4f}", lr=f"{current_lr:.1e}")

        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        writer.add_scalar('Loss/Adversarial', avg_loss, epoch)
        writer.add_image('Adversarial Patch', adversarial_patch, epoch)
        
        checkpoint_path = os.path.join(log_dir, CHECKPOINT_FILE)
        patch_filename = os.path.join(log_dir, f"epoch_{epoch:03d}_patch.png")
        patch_image = T.ToPILImage()(adversarial_patch.cpu())
        patch_image.save(patch_filename)
        
        print(f"\n--- Starting evaluation for epoch {epoch} ---")
        eval_model = YOLO(MODEL_NAME).to(device)
        # ✅ FIX: Pass the 'conf_threshold' argument to the evaluation function
        success_rate = run_evaluation(
            model=eval_model, 
            patch_path=patch_filename, 
            dataset_path=VALIDATION_DATASET_PATH, 
            conf_threshold=EVAL_CONF_THRESHOLD, 
            num_eval_images=num_eval_images, 
            seed=42
        )
        del eval_model
        torch.cuda.empty_cache()
        
        writer.add_scalar('Evaluation/SuccessRate', success_rate, epoch)
        print(f"  - Epoch {epoch} Evaluation Success Rate: {success_rate:.2f}% (Best: {max(best_success_rate, success_rate):.2f}%)")

        if success_rate > best_success_rate:
            best_success_rate = success_rate
            epochs_no_improve = 0
            print(f"  - ✅ New best success rate! Resetting patience counter.")
        else:
            epochs_no_improve += 1
            print(f"  - 📉 No improvement. Patience: {epochs_no_improve}/{EVAL_PATIENCE}")

        checkpoint = {'epoch': epoch, 'patch_state_dict': adversarial_patch.data.clone(), 'optimizer_state_dict': optimizer.state_dict(), 'scaler_state_dict': scaler.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'batch_size': batch_size, 'best_success_rate': best_success_rate, 'epochs_no_improve': epochs_no_improve}
        torch.save(checkpoint, checkpoint_path)
        print(f"--- Epoch {epoch} complete. Checkpoint and patch image saved. ---")
        
        if epochs_no_improve >= EVAL_PATIENCE:
            print(f"\n🛑 Early stopping triggered after {EVAL_PATIENCE} epochs with no improvement.")
            break

    writer.close()
    print("\nTraining run finished.")

# --- Crash Notification Function ---
def send_crash_notification(error_message):
    try:
        requests.post("https://ntfy.sh/PatchTraining", data=error_message.encode('utf-8'), headers={"Title": "Patch Training Script CRASHED", "Priority": "high", "Tags": "rotating_light,x"})
        print("\n🚨 Crash notification sent to ntfy.")
    except Exception as e:
        print(f"Failed to send crash notification: {e}")

# --- Main execution block ---
if __name__ == '__main__':
    def eval_images_type(value):
        if value.lower() == 'all': return -1
        try:
            ivalue = int(value)
            if ivalue <= 0 and ivalue != -1: raise argparse.ArgumentTypeError(f"invalid positive int value: '{value}'")
            return ivalue
        except ValueError:
            raise argparse.ArgumentTypeError(f"invalid value: '{value}', must be an integer or 'all'")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu': print("⚠️ CUDA not available. Running on CPU. This will be very slow.")

    parser = argparse.ArgumentParser(description="Train adversarial patches against a YOLO model.")
    parser.add_argument('--batch_size', type=int, default=8, help='Set the training batch size.')
    parser.add_argument('--autotune', action='store_true', help='Automatically find the best batch size.')
    parser.add_argument('--resume', type=str, default=None, help=f'Path to checkpoint to resume a specific run.')
    parser.add_argument('--starter_image', type=str, default=None, help='Path to an image to use as the starting patch.')
    parser.add_argument('--num_eval_images', type=eval_images_type, default='all', help="Number of images for evaluation, or 'all'.")
    parser.add_argument('--no_tensorboard', action='store_true', help='Disable TensorBoard logging and launch.')
    args = parser.parse_args()

    if not args.no_tensorboard:
        parent_log_dir = "runs"
        tb = program.TensorBoard(); tb.configure(argv=[None, '--logdir', parent_log_dir]); url = tb.launch()
        print("\n" + "="*60 + f"\n📈 TensorBoard is running: {url}\n   (It will show all runs from all sessions)\n" + "="*60 + "\n")

    if args.starter_image and args.resume: print("Error: --starter_image and --resume cannot be used together."); sys.exit(1)

    final_batch_size, resume_from = args.batch_size, args.resume
    
    if not args.resume and args.autotune:
        if device.type == 'cuda':
            temp_model = YOLO(MODEL_NAME).to(device); temp_model.model.train()
            final_batch_size = autotune_batch_size(device, temp_model, DummyDataset())
            del temp_model; torch.cuda.empty_cache()
        else:
            print("Autotune is only available for GPU (CUDA). Using default batch size.")
    
    session_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    session_log_dir = os.path.join("runs", session_timestamp)

    if resume_from:
        if not os.path.exists(resume_from): print(f"Error: Checkpoint file not found: '{resume_from}'"); sys.exit(1)
        checkpoint = torch.load(resume_from, map_location=device)
        final_batch_size = checkpoint.get('batch_size', args.batch_size)
        print(f"Resuming with batch size {final_batch_size} from checkpoint.")
        session_log_dir = os.path.dirname(resume_from)

    final_learning_rate = (final_batch_size / BASE_BATCH_SIZE) * BASE_LR

    os.makedirs(session_log_dir, exist_ok=True)

    try:
        if not os.path.exists(DATASET_PATH):
            print(f"Error: Training dataset path not found: '{DATASET_PATH}'")
        elif not os.path.exists(VALIDATION_DATASET_PATH):
            print(f"Error: Validation dataset path not found: '{VALIDATION_DATASET_PATH}'")
        else:
            train_adversarial_patch(
                batch_size=final_batch_size, 
                learning_rate=final_learning_rate,
                log_dir=session_log_dir,
                device=device,
                num_eval_images=args.num_eval_images,
                use_tensorboard=(not args.no_tensorboard),
                resume_path=resume_from,
                starter_image_path=args.starter_image
            )
    except Exception as e:
        print("\n" + "="*60 + f"\n💥 An unexpected error occurred during training! Sending notification...\n" + "="*60)
        error_info = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        send_crash_notification(error_info)
        raise e
            
    print("\n" + "="*60 + "\n✅ Training session completed.\n" + "="*60)
