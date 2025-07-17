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
import multiprocessing as mp
import time
import json

# Import the evaluation logic from the user-provided file
from evaluation_logic import run_evaluation

# --- Configuration (Defaults) ---
DATASET_PATH = 'VisDrone2019-DET-train'
VALIDATION_DATASET_PATH = 'VisDrone2019-DET-val'
MODEL_NAME = 'yolov11n.pt'
PATCH_RESOLUTION = 100

# --- Base values for dynamic learning rate scaling ---
BASE_LR = 0.0001
BASE_BATCH_SIZE = 8.0

# --- Smart Training Config ---
PLATEAU_PATIENCE = 5
CHECKPOINT_FILE = "patch_checkpoint.pth"

# --- Early Stopping Configuration ---
EVAL_PATIENCE = 10 
EVAL_CONF_THRESHOLD = 0.25

# --- Logging and Process Setup ---
def setup_process_logging(log_file_path):
    """Redirects stdout and stderr of a child process to a log file."""
    sys.stdout = open(log_file_path, 'a', buffering=1)
    sys.stderr = sys.stdout
    # The tqdm progress bar instance itself will be directed to the correct output stream.

# --- Define collate_fn at the top level so it can be pickled ---
def custom_collate_fn(batch):
    """
    Custom collate function to handle batches where some images might not have annotations.
    It filters out None items that might result from a failed __getitem__ call.
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return None, None
    
    images, boxes = zip(*batch)
    return torch.stack(images, 0), boxes

# --- Dataset Classes ---
class VisDroneDataset(Dataset):
    """Loads images AND their corresponding bounding box annotations."""
    def __init__(self, root_dir, transform=None):
        self.root_dir, self.transform = root_dir, transform
        self.image_dir = os.path.join(root_dir, 'images')
        self.annotation_dir = os.path.join(root_dir, 'annotations_v11')
        self.image_files = sorted(os.listdir(self.image_dir))

    def __len__(self): return len(self.image_files)
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
        except (IOError, OSError):
            print(f"Warning: Could not read image {img_path}. Skipping.")
            return None

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
                    except (ValueError, IndexError): continue
        boxes = torch.tensor(boxes, dtype=torch.float32)
        if self.transform: image = self.transform(image)
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
def autotune_batch_size(args_dict):
    """Finds the largest batch size that fits in a specific GPU's VRAM. Designed to be run in a process."""
    gpu_id = args_dict['gpu_id']
    log_file = args_dict['log_file']
    setup_process_logging(log_file) # Redirect output for this process

    device = torch.device(f"cuda:{gpu_id}")
    model = YOLO(MODEL_NAME).to(device)
    model.model.train()
    dataset = DummyDataset()
    batch_size = 2
    
    print(f"🚀 Autotuning GPU {gpu_id} from batch size {batch_size}...")
    while True:
        try:
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            images = next(iter(dataloader)).to(device)
            dummy_patch = torch.rand((3, PATCH_RESOLUTION, PATCH_RESOLUTION), device=device, requires_grad=True)
            for j in range(min(images.size(0), batch_size)):
                 patch_side = random.randint(50, 150)
                 resized_patch = TF.resize(dummy_patch, [patch_side, patch_side], antialias=True)
                 x, y = random.randint(0, 640 - patch_side), random.randint(0, 640 - patch_side)
                 images[j, :, y:y+patch_side, x:x+patch_side] = resized_patch
            with torch.amp.autocast(device.type):
                raw_preds = model.model(images)[0].transpose(1, 2)
                loss = torch.mean(torch.max(raw_preds[..., 4:], dim=-1)[0])
            loss.backward()
            print(f"  - GPU {gpu_id}: Batch size {batch_size} fits.")
            del images, raw_preds, loss, dummy_patch
            batch_size *= 2
        except torch.cuda.OutOfMemoryError:
            max_size = batch_size // 2
            print(f"  - ⚠️ GPU {gpu_id}: OOM at {batch_size}. Optimal: {max_size}.")
            torch.cuda.empty_cache()
            return max_size
        except StopIteration:
            print(f"  - ✅ GPU {gpu_id}: Batch size {batch_size} fits, but dataset is too small to double.")
            return batch_size

# --- Dummy Writer for Headless Mode ---
class DummyWriter:
    def add_scalar(self, *args, **kwargs): pass
    def add_image(self, *args, **kwargs): pass
    def close(self): pass

# --- Adversarial Patch Training (Now runs as a separate process) ---
def train_adversarial_patch(args_dict):
    """Main training function, designed to be run in a separate process."""
    log_file = args_dict['log_file']
    setup_process_logging(log_file)

    gpu_id = args_dict['gpu_id']
    batch_size = args_dict['batch_size']
    learning_rate = args_dict['learning_rate']
    log_dir = args_dict['log_dir']
    num_eval_images = args_dict['num_eval_images']
    use_tensorboard = args_dict['use_tensorboard']
    starter_image_path = args_dict['starter_image_path']
    status_queue = args_dict['status_queue']
    is_parallel = args_dict['is_parallel']

    device = torch.device(f"cuda:{gpu_id}")
    if device.type == 'cuda': cudnn.benchmark = True
    
    writer = SummaryWriter(log_dir=log_dir) if use_tensorboard else DummyWriter()
    
    training_model = YOLO(MODEL_NAME).to(device)
    training_model.model.train()

    if device.type == 'cuda' and not is_parallel:
        try:
            training_model.model = torch.compile(training_model.model)
            print("✅ Model compiled successfully with torch.compile().")
        except Exception: 
            print("⚠️ torch.compile() failed. Running without compilation.")

    if starter_image_path and os.path.exists(starter_image_path):
        starter_image = Image.open(starter_image_path).convert("RGB")
        transform_starter = T.Compose([T.Resize((PATCH_RESOLUTION, PATCH_RESOLUTION)), T.ToTensor()])
        adversarial_patch = transform_starter(starter_image).to(device)
        adversarial_patch.requires_grad_(True)
    else:
        adversarial_patch = torch.rand((3, PATCH_RESOLUTION, PATCH_RESOLUTION), device=device, requires_grad=True)

    optimizer = torch.optim.Adam([adversarial_patch], lr=learning_rate)
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == 'cuda'))
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=PLATEAU_PATIENCE)

    epoch, best_success_rate, epochs_no_improve = 0, -1.0, 0

    transform = T.Compose([T.Resize((640, 640)), T.ToTensor()])
    dataset = VisDroneDataset(root_dir=DATASET_PATH, transform=transform)
    num_workers = os.cpu_count() // (2 * torch.cuda.device_count()) if device.type == 'cuda' and torch.cuda.device_count() > 0 else 4
    pin_memory = (device.type == 'cuda')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, collate_fn=custom_collate_fn)

    while True:
        epoch += 1
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"GPU {gpu_id} | Epoch {epoch}", leave=False, position=gpu_id, file=sys.__stdout__)

        for i, (images, gt_boxes_batch) in enumerate(progress_bar):
            if images is None: continue
            
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
            
            status_queue.put({'gpu_id': gpu_id, 'epoch': epoch, 'progress': f"{i+1}/{len(dataloader)}", 'loss': f"{total_loss/(i+1):.4f}", 'best_rate': f"{best_success_rate:.2f}%", 'patience': f"{epochs_no_improve}/{EVAL_PATIENCE}"})

        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        writer.add_scalar('Loss/Adversarial', avg_loss, epoch)
        writer.add_image('Adversarial Patch', adversarial_patch, epoch)
        
        patch_filename = os.path.join(log_dir, f"epoch_{epoch:03d}_patch.png")
        patch_image = T.ToPILImage()(adversarial_patch.cpu())
        patch_image.save(patch_filename)
        
        eval_model = YOLO(MODEL_NAME).to(device)
        success_rate = run_evaluation(model=eval_model, patch_path=patch_filename, dataset_path=VALIDATION_DATASET_PATH, conf_threshold=EVAL_CONF_THRESHOLD, num_eval_images=num_eval_images, seed=42)
        del eval_model; torch.cuda.empty_cache()
        
        writer.add_scalar('Evaluation/SuccessRate', success_rate, epoch)

        if success_rate > best_success_rate:
            best_success_rate = success_rate
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        checkpoint = {'epoch': epoch, 'patch_state_dict': adversarial_patch.data.clone(), 'optimizer_state_dict': optimizer.state_dict(), 'scaler_state_dict': scaler.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'batch_size': batch_size, 'best_success_rate': best_success_rate, 'epochs_no_improve': epochs_no_improve}
        torch.save(checkpoint, os.path.join(log_dir, CHECKPOINT_FILE))
        
        if epochs_no_improve >= EVAL_PATIENCE:
            status_queue.put({'gpu_id': gpu_id, 'status': 'stopped', 'best_rate': f"{best_success_rate:.2f}%", 'epoch': epoch})
            break

    writer.close()
    status_queue.put({'gpu_id': gpu_id, 'status': 'finished', 'best_rate': f"{best_success_rate:.2f}%", 'epoch': epoch})

# --- Crash Notification Function ---
def send_crash_notification(error_message):
    try:
        requests.post("https://ntfy.sh/PatchTraining", data=error_message.encode('utf-8'), headers={"Title": "Patch Training Script CRASHED", "Priority": "high", "Tags": "rotating_light,x"})
    except Exception: pass

# --- Main execution block ---
if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass 

    def eval_images_type(value):
        if value.lower() == 'all': return -1
        try:
            ivalue = int(value)
            if ivalue <= 0 and ivalue != -1: raise argparse.ArgumentTypeError(f"invalid positive int value: '{value}'")
            return ivalue
        except ValueError:
            raise argparse.ArgumentTypeError(f"invalid value: '{value}', must be an integer or 'all'")

    parser = argparse.ArgumentParser(description="Train adversarial patches against a YOLO model.")
    parser.add_argument('--batch_size', type=int, default=8, help='Set the training batch size PER GPU.')
    parser.add_argument('--autotune', action='store_true', help='Automatically find the best batch size PER GPU.')
    parser.add_argument('--starter_image', type=str, default=None, help='Path to an image to use as the starting patch.')
    parser.add_argument('--num_eval_images', type=eval_images_type, default=100, help="Number of images for evaluation, or 'all'.")
    parser.add_argument('--no_tensorboard', action='store_true', help='Disable TensorBoard logging and launch.')
    parser.add_argument('--parallel', action='store_true', help='Run one training process per available GPU in parallel.')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count() if device.type == 'cuda' else 0

    if device.type == 'cpu': print("⚠️ CUDA not available. Running on CPU. This will be very slow.")
    if args.parallel and num_gpus < 2: print("⚠️ --parallel flag requires at least 2 GPUs. Running in single GPU mode."); args.parallel = False

    if not args.no_tensorboard:
        parent_log_dir = "runs"
        tb = program.TensorBoard(); tb.configure(argv=[None, '--logdir', parent_log_dir]); url = tb.launch()
        print("\n" + "="*60 + f"\n📈 TensorBoard is running: {url}\n" + "="*60 + "\n")

    session_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    session_log_dir = os.path.join("runs", session_timestamp)
    os.makedirs(session_log_dir, exist_ok=True)

    # --- Parallel Execution Logic ---
    if args.parallel and num_gpus > 1:
        batch_sizes = []
        if args.autotune:
            autotune_args = [{'gpu_id': i, 'log_file': os.path.join(session_log_dir, f"autotune_gpu_{i}.log")} for i in range(num_gpus)]
            with mp.Pool(processes=num_gpus) as pool:
                results = pool.map(autotune_batch_size, autotune_args)
                batch_sizes = results
        else:
            batch_sizes = [args.batch_size] * num_gpus

        processes, final_results = [], []
        status_queue = mp.Queue()
        
        for i in range(num_gpus):
            run_log_dir = os.path.join(session_log_dir, f"GPU_{i}_Run")
            os.makedirs(run_log_dir, exist_ok=True)
            lr = (batch_sizes[i] / BASE_BATCH_SIZE) * BASE_LR
            process_args = {
                'gpu_id': i, 'batch_size': batch_sizes[i], 'learning_rate': lr,
                'log_dir': run_log_dir, 'num_eval_images': args.num_eval_images,
                'use_tensorboard': (not args.no_tensorboard),
                'starter_image_path': args.starter_image, 'status_queue': status_queue,
                'is_parallel': True, 'log_file': os.path.join(run_log_dir, 'output.log')
            }
            p = mp.Process(target=train_adversarial_patch, args=(process_args,))
            processes.append(p)
            p.start()

        statuses = {i: {} for i in range(num_gpus)}
        active_processes = num_gpus
        while active_processes > 0:
            try:
                update = status_queue.get(timeout=1)
                gpu_id = update['gpu_id']
                statuses[gpu_id] = update
                if statuses[gpu_id].get('status') in ['stopped', 'finished']:
                    active_processes -= 1
                
                os.system('cls' if os.name == 'nt' else 'clear')
                print(f"--- Live Parallel Training Status ({datetime.now().strftime('%H:%M:%S')}) ---")
                print("{:<6} {:<10} {:<8} {:<12} {:<12} {:<12} {:<10}".format("GPU", "Status", "Epoch", "Progress", "Adv Loss", "Best Rate", "Patience"))
                print("-" * 80)
                for i in range(num_gpus):
                    s = statuses.get(i, {})
                    status = s.get('status', 'Starting')
                    print("{:<6} {:<10} {:<8} {:<12} {:<12} {:<12} {:<10}".format(
                        i, status.capitalize(), s.get('epoch', '-'), s.get('progress', '-'), s.get('loss', '-'), s.get('best_rate', '-'), s.get('patience', '-')))
                time.sleep(1)
            except:
                if all(not p.is_alive() for p in processes): break
        
        for p in processes: p.join()

    # --- Single GPU Execution Logic ---
    else:
        final_batch_size = args.batch_size
        if args.autotune and device.type == 'cuda':
            final_batch_size = autotune_batch_size({'gpu_id': 0, 'log_file': os.path.join(session_log_dir, 'autotune.log')})

        final_learning_rate = (final_batch_size / BASE_BATCH_SIZE) * BASE_LR
        
        try:
            if not os.path.exists(DATASET_PATH): print(f"Error: Training dataset path not found: '{DATASET_PATH}'")
            elif not os.path.exists(VALIDATION_DATASET_PATH): print(f"Error: Validation dataset path not found: '{VALIDATION_DATASET_PATH}'")
            else:
                dummy_queue = type('Queue', (), {'put': lambda self, x: None})()
                train_args = {
                    'gpu_id': 0, 'batch_size': final_batch_size, 'learning_rate': final_learning_rate,
                    'log_dir': session_log_dir, 'num_eval_images': args.num_eval_images,
                    'use_tensorboard': (not args.no_tensorboard), 'resume_path': args.resume,
                    'starter_image_path': args.starter_image, 'status_queue': dummy_queue,
                    'is_parallel': False, 'log_file': os.path.join(session_log_dir, 'output.log')
                }
                train_adversarial_patch(train_args)
        except Exception as e:
            print("\n" + "="*60 + f"\n💥 An unexpected error occurred during training! Sending notification...\n" + "="*60)
            error_info = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            send_crash_notification(error_info)
            raise e
            
    print("\n" + "="*60 + "\n✅ Training session completed.\n" + "="*60)
