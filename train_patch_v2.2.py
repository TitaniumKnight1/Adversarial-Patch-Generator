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
from tqdm import tqdm
import argparse
import torch.backends.cudnn as cudnn
from tensorboard import program
import requests
import traceback
import sys
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime
import math

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

# --- ANSI Color Codes for Rich CLI Output ---
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

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
                        if len(parts) >= 4:
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

# --- Loss Functions ---
class TotalVariationLoss(torch.nn.Module):
    """Calculates the total variation of an image, encouraging smoothness."""
    def __init__(self):
        super(TotalVariationLoss, self).__init__()

    def forward(self, patch):
        if patch.dim() == 3:
            patch = patch.unsqueeze(0)
        wh_diff = torch.sum(torch.abs(patch[:, :, :, 1:] - patch[:, :, :, :-1]))
        ww_diff = torch.sum(torch.abs(patch[:, :, 1:, :] - patch[:, :, :-1, :]))
        return (wh_diff + ww_diff) / (patch.size(2) * patch.size(3))

# --- Autotune Function ---
def autotune_batch_size(device, model, dataset, initial_batch_size=2):
    batch_size = initial_batch_size
    print(f"🚀 {bcolors.HEADER}Starting batch size autotune from size {batch_size}...{bcolors.ENDC}")
    while True:
        try:
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            images = next(iter(dataloader)).to(device)
            dummy_patch = torch.rand((3, PATCH_SIZE, PATCH_SIZE), device=device, requires_grad=True)
            optimizer = torch.optim.Adam([dummy_patch], lr=0.01)

            for j in range(min(images.size(0), batch_size)):
                 x, y = random.randint(0, 640 - PATCH_SIZE), random.randint(0, 640 - PATCH_SIZE)
                 images[j, :, x:x+PATCH_SIZE, y:y+PATCH_SIZE] = dummy_patch

            with torch.amp.autocast(device.type):
                raw_preds = model.model(images)[0].transpose(1, 2)
                loss = torch.mean(torch.max(raw_preds[..., 4:], dim=-1)[0])

            loss.backward()
            optimizer.zero_grad(set_to_none=True)
            print(f"✅ {bcolors.OKGREEN}Batch size {batch_size} fits in memory. Trying next size...{bcolors.ENDC}")
            del images, raw_preds, loss, dummy_patch, optimizer
            batch_size *= 2
            
        except torch.cuda.OutOfMemoryError:
            max_size = batch_size // 2
            print(f"⚠️ {bcolors.WARNING}OOM at {batch_size}. Optimal batch size set to: {max_size}{bcolors.ENDC}")
            torch.cuda.empty_cache()
            return max_size
        except StopIteration:
            print(f"✅ {bcolors.OKGREEN}Batch size {batch_size} fits, but dataset is too small to double.{bcolors.ENDC}")
            return batch_size

# --- Adversarial Patch Training ---
def train_adversarial_patch(batch_size, learning_rate, log_dir, max_epochs, device, tv_weight, resume_path=None, starter_image_path=None):
    if device.type == 'cuda': cudnn.benchmark = True
    
    writer = SummaryWriter(log_dir=log_dir)
    model = YOLO(MODEL_NAME)
    model.to(device)
    
    # --- Multi-GPU Preparation ---
    # If you have multiple GPUs, you can wrap the model with DataParallel.
    # The script will automatically use all available GPUs for inference.
    if torch.cuda.device_count() > 1:
        print(f"🚀 {bcolors.HEADER}Using {torch.cuda.device_count()} GPUs!{bcolors.ENDC}")
        model.model = torch.nn.DataParallel(model.model)

    model.model.train()

    if device.type == 'cuda':
        try:
            # Note: torch.compile is not compatible with DataParallel in some versions.
            # If using DataParallel, you might need to disable compile.
            if not isinstance(model.model, torch.nn.DataParallel):
                model.model = torch.compile(model.model)
                print(f"✅ {bcolors.OKGREEN}Model compiled successfully with torch.compile().{bcolors.ENDC}")
        except Exception as e:
            print(f"⚠️ {bcolors.WARNING}torch.compile() failed: {e}. Running without compilation.{bcolors.ENDC}")

    if starter_image_path and os.path.exists(starter_image_path):
        print(f"🌱 {bcolors.OKCYAN}Initializing patch from starter image: {starter_image_path}{bcolors.ENDC}")
        starter_image = Image.open(starter_image_path).convert("RGB")
        transform_starter = T.Compose([T.Resize((PATCH_SIZE, PATCH_SIZE)), T.ToTensor()])
        adversarial_patch = transform_starter(starter_image).to(device)
        adversarial_patch.requires_grad_(True)
    else:
        print(f"🎨 {bcolors.OKCYAN}Initializing patch with random noise.{bcolors.ENDC}")
        adversarial_patch = torch.rand((3, PATCH_SIZE, PATCH_SIZE), device=device, requires_grad=True)

    optimizer = torch.optim.Adam([adversarial_patch], lr=learning_rate, amsgrad=True)
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=PLATEAU_PATIENCE, verbose=True)
    total_variation = TotalVariationLoss().to(device)

    start_epoch = 0
    if resume_path and os.path.exists(resume_path):
        print(f"🔄 {bcolors.OKBLUE}Resuming training from checkpoint: {resume_path}{bcolors.ENDC}")
        checkpoint = torch.load(resume_path, map_location=device)
        adversarial_patch.data = checkpoint['patch_state_dict'].to(device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scaler_state_dict' in checkpoint: scaler.load_state_dict(checkpoint['scaler_state_dict'])
        if 'scheduler_state_dict' in checkpoint: scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resumed from epoch {start_epoch}.")

    transform = T.Compose([T.Resize((640, 640)), T.ToTensor()])
    dataset = VisDroneDataset(root_dir=DATASET_PATH, transform=transform)
    num_workers = min(os.cpu_count() // 2, 16) if os.cpu_count() else 4
    pin_memory = (device.type == 'cuda')
    def collate_fn(batch):
        images = [item[0] for item in batch]
        boxes = [item[1] for item in batch]
        return torch.stack(images, 0), boxes

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn)

    print(f"\n🚀 {bcolors.BOLD}Starting Adversarial Patch Training{bcolors.ENDC}")
    print(f"   - Device: {bcolors.OKCYAN}{device.type.upper()}{bcolors.ENDC}")
    print(f"   - Batch Size: {bcolors.OKCYAN}{batch_size}{bcolors.ENDC}")
    print(f"   - Scaled LR: {bcolors.OKCYAN}{learning_rate:.2e}{bcolors.ENDC}")
    print(f"   - DataLoaders: {bcolors.OKCYAN}{num_workers}{bcolors.ENDC}")
    print(f"   - TV Weight: {bcolors.OKCYAN}{tv_weight}{bcolors.ENDC}")
    print(f"   - Early Stopping Patience: {bcolors.OKCYAN}{EARLY_STOPPING_PATIENCE}{bcolors.ENDC}")

    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(start_epoch, max_epochs):
        epoch_start_time = time.time()
        total_adv_loss, total_tv_loss = 0, 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{max_epochs}", ncols=120)

        for i, (images, gt_boxes_batch) in enumerate(progress_bar):
            images = images.to(device, non_blocking=pin_memory)
            
            angle, scale = random.uniform(-15, 15), random.uniform(0.8, 1.2)
            transformed_patch = TF.rotate(adversarial_patch, angle)
            new_size = int(PATCH_SIZE * scale)
            transformed_patch = TF.resize(transformed_patch, (new_size, new_size))
            transformed_patch = TF.center_crop(transformed_patch, (PATCH_SIZE, PATCH_SIZE))
            transformed_patch = TF.adjust_brightness(transformed_patch, random.uniform(0.7, 1.3))
            transformed_patch = TF.adjust_contrast(transformed_patch, random.uniform(0.7, 1.3))
            transformed_patch.data.clamp_(0,1)

            for img_idx in range(images.size(0)):
                gt_boxes = gt_boxes_batch[img_idx]
                if len(gt_boxes) > 0:
                    box_idx = random.randint(0, len(gt_boxes) - 1)
                    box = gt_boxes[box_idx].int()
                    x1, y1, x2, y2 = box
                    center_x, center_y = random.randint(x1, x2), random.randint(y1, y2)
                    patch_x = max(0, min(center_x - PATCH_SIZE // 2, images.shape[3] - PATCH_SIZE))
                    patch_y = max(0, min(center_y - PATCH_SIZE // 2, images.shape[2] - PATCH_SIZE))
                    images[img_idx, :, patch_y:patch_y+PATCH_SIZE, patch_x:patch_x+PATCH_SIZE] = transformed_patch
                else:
                    x_start, y_start = random.randint(0, 640 - PATCH_SIZE), random.randint(0, 640 - PATCH_SIZE)
                    images[img_idx, :, y_start:y_start+PATCH_SIZE, x_start:x_start+PATCH_SIZE] = transformed_patch

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
            current_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix(
                adv_loss=f"{(total_adv_loss)/(i+1):.4f}", 
                tv_loss=f"{(total_tv_loss)/(i+1):.4f}",
                lr=f"{current_lr:.1e}",
                best_loss=f"{best_loss:.4f}"
            )

        avg_adv_loss = total_adv_loss / len(dataloader)
        avg_tv_loss = total_tv_loss / len(dataloader)
        avg_total_loss = avg_adv_loss + tv_weight * avg_tv_loss
        epoch_duration = time.time() - epoch_start_time
        
        print("\n" + "="*80)
        print(f"{bcolors.BOLD}Epoch {epoch + 1} Summary{bcolors.ENDC} | {bcolors.WARNING}Patience: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}{bcolors.ENDC}")
        print(f"  🕒 Time: {bcolors.OKCYAN}{epoch_duration:.2f}s{bcolors.ENDC} | 📉 Adv Loss: {bcolors.OKGREEN}{avg_adv_loss:.4f}{bcolors.ENDC} | 🎨 TV Loss: {bcolors.OKGREEN}{avg_tv_loss:.4f}{bcolors.ENDC} | 🔥 Total Loss: {bcolors.OKGREEN}{avg_total_loss:.4f}{bcolors.ENDC} | 💡 LR: {bcolors.OKCYAN}{current_lr:.1e}{bcolors.ENDC}")
        print("="*80 + "\n")

        scheduler.step(avg_total_loss)
        writer.add_scalar('Loss/Adversarial', avg_adv_loss, epoch)
        writer.add_scalar('Loss/TotalVariation', avg_tv_loss, epoch)
        writer.add_scalar('Loss/Total', avg_total_loss, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        writer.add_image('Adversarial Patch', adversarial_patch, epoch)
        
        checkpoint = {'epoch': epoch + 1, 'patch_state_dict': adversarial_patch.data.clone(), 'optimizer_state_dict': optimizer.state_dict(), 'scaler_state_dict': scaler.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'batch_size': batch_size}
        torch.save(checkpoint, os.path.join(log_dir, CHECKPOINT_FILE))
        
        if avg_total_loss < best_loss:
            print(f"🎉 {bcolors.OKGREEN}New best loss found: {avg_total_loss:.4f} (previously {best_loss:.4f}). Saving best patch.{bcolors.ENDC}")
            best_loss = avg_total_loss
            epochs_no_improve = 0
            torch.save(checkpoint, os.path.join(log_dir, BEST_CHECKPOINT_FILE))
            patch_image = T.ToPILImage()(adversarial_patch.cpu())
            patch_image.save(os.path.join(log_dir, "best_patch.png"))
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"\n🛑 {bcolors.FAIL}{bcolors.BOLD}Early stopping triggered after {EARLY_STOPPING_PATIENCE} epochs with no improvement.{bcolors.ENDC}")
            print(f"   Best loss achieved: {best_loss:.4f}")
            break

    writer.close()
    print(f"\n🎉 {bcolors.OKGREEN}{bcolors.BOLD}Training run finished.{bcolors.ENDC}")

# --- Crash Notification Function ---
def send_crash_notification(error_message):
    try:
        requests.post("https://ntfy.sh/PatchTraining", data=error_message.encode('utf-8'), headers={"Title": "Patch Training Script CRASHED", "Priority": "high", "Tags": "rotating_light,x"})
        print(f"\n🚨 {bcolors.FAIL}Crash notification sent to ntfy.{bcolors.ENDC}")
    except Exception as e:
        print(f"Failed to send crash notification: {e}")

# --- Main execution block ---
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        print(f"⚠️ {bcolors.WARNING}CUDA not available. Running on CPU. This will be very slow.{bcolors.ENDC}")

    parent_log_dir = "runs"
    if not os.path.exists(parent_log_dir): os.makedirs(parent_log_dir)
    tb = program.TensorBoard(); tb.configure(argv=[None, '--logdir', parent_log_dir]); url = tb.launch()
    print("\n" + "="*60 + f"\n📈 {bcolors.BOLD}TensorBoard is running: {bcolors.OKBLUE}{url}{bcolors.ENDC}\n" + "="*60 + "\n")

    parser = argparse.ArgumentParser(description="Train adversarial patches against a YOLO model with automatic LR scaling and early stopping.")
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume a specific run.')
    parser.add_argument('--starter_image', type=str, default=None, help='Path to an image to use as the starting point for the patch.')
    parser.add_argument('--tv_weight', type=float, default=1e-4, help='Weight for the Total Variation loss term.')
    parser.add_argument('--max_epochs', type=int, default=DEFAULT_MAX_EPOCHS, help='Maximum number of epochs to train for.')
    parser.add_argument('--early_stopping_patience', type=int, default=EARLY_STOPPING_PATIENCE, help='Number of epochs with no improvement to wait before stopping.')
    args = parser.parse_args()

    if args.starter_image and args.resume: 
        print(f"{bcolors.FAIL}Error: --starter_image and --resume cannot be used together.{bcolors.ENDC}"); sys.exit(1)

    EARLY_STOPPING_PATIENCE = args.early_stopping_patience
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        final_batch_size = checkpoint.get('batch_size', BASE_BATCH_SIZE)
        print(f"Resuming with batch size {final_batch_size} from checkpoint.")
    elif device.type == 'cuda':
        temp_model = YOLO(MODEL_NAME).to(device); temp_model.model.train()
        final_batch_size = autotune_batch_size(device, temp_model, DummyDataset())
        del temp_model; torch.cuda.empty_cache()
    else:
        print("Autotune is only for CUDA. Using base batch size.")
        final_batch_size = BASE_BATCH_SIZE

    scaled_lr = BASE_LEARNING_RATE * (final_batch_size / BASE_BATCH_SIZE)
    
    session_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(parent_log_dir, session_timestamp)
    
    if args.resume:
        log_dir = os.path.dirname(args.resume)

    try:
        if not os.path.exists(DATASET_PATH):
            print(f"{bcolors.FAIL}Error: Dataset path not found: '{DATASET_PATH}'{bcolors.ENDC}"); sys.exit(1)
        
        train_adversarial_patch(
            batch_size=final_batch_size, 
            learning_rate=scaled_lr,
            log_dir=log_dir,
            max_epochs=args.max_epochs,
            device=device,
            resume_path=args.resume,
            starter_image_path=args.starter_image,
            tv_weight=args.tv_weight
        )
    except Exception as e:
        print("\n" + "="*60 + f"\n💥 {bcolors.FAIL}{bcolors.BOLD}An unexpected error occurred! Sending notification...{bcolors.ENDC}\n" + "="*60)
        error_info = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        send_crash_notification(error_info)
        raise e
