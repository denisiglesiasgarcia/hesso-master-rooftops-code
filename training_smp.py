# 02_training_v21.py
import argparse
import csv
import datetime
import gc
import json
import os
import time
import fcntl
from pathlib import Path


import albumentations as A
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from PIL import Image
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from training_configs import CONFIGS
import psutil
import tempfile
import shutil
import atexit

matplotlib.use("Agg")

# Ampere
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

# ====================================
# Constants - Hardcoded Paths
# ====================================

BASE_DATASET_DIR = "datasets/supervisely/dataset_processed_20250523-173715"
DATASET_IMAGES_DIR = BASE_DATASET_DIR + "/images"
DATASET_MASKS_DIR = BASE_DATASET_DIR + "/masks"
TEST_DATASET_FILE = BASE_DATASET_DIR + "/test_dataset.txt"

SCRIPT_START_TIME = time.time()

# ====================================
# Memory Management Utilities
# ====================================
def get_memory_usage(device, stage=""):
    """Format memory information for logging."""
    # Currently allocated to tensors
    allocated = torch.cuda.memory_allocated(device)
    # Total reserved by PyTorch (includes cache)
    reserved = torch.cuda.memory_reserved(device)
    # Peak allocated since last reset
    max_allocated = torch.cuda.max_memory_allocated(device)
    
    total = torch.cuda.get_device_properties(device).total_memory
    
    allocated_gb = allocated / 1024**3
    reserved_gb = reserved / 1024**3
    max_allocated_gb = max_allocated / 1024**3
    total_gb = total / 1024**3
    
    allocated_pct = (allocated / total) * 100
    reserved_pct = (reserved / total) * 100
    max_allocated_pct = (max_allocated / total) * 100
    
    info = (
        f"Memory {stage}: "
        f"Allocated: {allocated_gb:.1f}GB ({allocated_pct:.1f}%) | "
        f"Reserved: {reserved_gb:.1f}GB ({reserved_pct:.1f}%) | "
        f"Peak: {max_allocated_gb:.1f}GB ({max_allocated_pct:.1f}%) | "
        f"Total: {total_gb:.1f}GB"
    )
    
    return info, reserved_pct, max_allocated_pct

def clear_memory():
    """Clear GPU memory cache and perform garbage collection."""
    if torch.cuda.is_available():
        for _ in range(3):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            time.sleep(0.01)


# ====================================
# Logger Setup
# ====================================
def get_elapsed_time():
    """Get elapsed time since script start in HH:MM:SS format."""
    elapsed = time.time() - SCRIPT_START_TIME
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def elapsed_time_filter(record):
    """Add elapsed time to log record."""
    record["extra"]["elapsed"] = get_elapsed_time()
    return True


def setup_logger(log_dir, verbose=True):
    """Setup loguru logger with elapsed time tracking and improved formatting."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "training.log")
    detailed_log_file = os.path.join(log_dir, "training_detailed.log")

    # Remove default handler
    logger.remove()

    # Define formats with elapsed time
    if verbose:
        console_format = "{time:HH:mm:ss} | +{extra[elapsed]} | {level: <4} | {function: <15} | {message}"
        console_filter = elapsed_time_filter
    else:
        console_format = "{time:HH:mm:ss} | +{extra[elapsed]} | {message}"
        console_filter = lambda record: (
            elapsed_time_filter(record)
            and (
                # Include all ERROR and WARNING level messages
                record["level"].name in ["ERROR", "WARNING"]
                or
                # Include messages with specific keywords
                any(
                    keyword in record["message"]
                    for keyword in [
                        "STARTING",
                        "COMPLETED",
                        "Epoch",
                        "Best",
                        "PROGRESS",
                        "FINAL",
                        "Configuration",
                        "Training",
                        "Statistics",
                        "Total samples",
                        "IoU range",
                        "Mean:",
                        "Median:",
                        "Samples with IoU",
                        "Worst 5",
                        "Best 5",
                        "NaN",
                        "Inf",
                        "VALIDATION ERROR",
                    ]
                )
            )
        )

    # Console handler - simplified
    logger.add(
        lambda msg: print(msg, end=""),  # Keep your existing approach since it works
        level="INFO",
        format=console_format,
        colorize=True,  # Enable colors for better readability
        filter=console_filter,
    )

    # Main log file with elapsed time
    logger.add(
        log_file,
        rotation="50 MB",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | +{extra[elapsed]} | {level: <8} | {message}",
        colorize=False,
        filter=lambda record: (
            elapsed_time_filter(record)
            and not any(
                keyword in record["message"]
                for keyword in [
                    "debug_dataset_samples",
                    "Original image size",
                    "Original mask",
                    "Processed",
                ]
            )
        ),
    )

    # Detailed log file with all information
    logger.add(
        detailed_log_file,
        rotation="200 MB",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | +{extra[elapsed]} | {level: <8} | {name}:{function}:{line} | {message}",
        colorize=False,
        filter=elapsed_time_filter,
    )

    # Log startup information
    logger.info(f"Logger initialized. Main: {log_file}, Detailed: {detailed_log_file}")
    logger.info(
        f"Script started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(SCRIPT_START_TIME))}"
    )
    return logger


# ====================================
# Dataset Class
# ====================================
class SimpleSegmentationDataset(Dataset):
    """Memory-optimized dataset class."""

    def __init__(
        self, image_paths, mask_paths, img_size, transform=None, cache_size=250
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_size = img_size
        self.transform = transform
        self.cache_size = cache_size
        self._cache = {}

        # Validate paths
        assert len(image_paths) == len(mask_paths), "Mismatch between images and masks"

        # Check if files exist
        valid_pairs = []
        for img_path, mask_path in zip(image_paths, mask_paths):
            if os.path.exists(img_path) and os.path.exists(mask_path):
                valid_pairs.append((img_path, mask_path))
            else:
                logger.warning(f"Missing files: {img_path} or {mask_path}")

        self.image_paths = [pair[0] for pair in valid_pairs]
        self.mask_paths = [pair[1] for pair in valid_pairs]

        logger.info(f"Dataset initialized with {len(valid_pairs)} valid pairs")

    def __len__(self):
        return len(self.image_paths)

    def _load_and_process(self, idx):
        """Optimized load and process with OpenCV."""
        try:
            # Use OpenCV for faster loading (30% faster than PIL)
            image = cv2.imread(self.image_paths[idx])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.img_size, interpolation=cv2.INTER_LINEAR)
            
            # Load mask with OpenCV too
            mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
            
            # Create binary mask
            if mask.max() > 1:
                mask = mask.astype(np.float32) / 255.0
                mask = (mask > 0.5).astype(np.uint8)
            else:
                mask = (mask > 0).astype(np.uint8)
                
            return image, mask
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {e}")
            raise e

    def __getitem__(self, idx):
        # Simple caching for frequently accessed items
        if idx in self._cache:
            image, mask = self._cache[idx]
        else:
            image, mask = self._load_and_process(idx)

            # Add to cache if not full
            if len(self._cache) < self.cache_size:
                self._cache[idx] = (image.copy(), mask.copy())

        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Convert to tensors with memory-efficient dtypes
        image = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)
        mask = torch.from_numpy(mask.astype(np.float32))

        return image, mask


# ====================================
# Data Augmentation
# ====================================
def get_transforms(is_training=True):
    """
    Get data augmentation transforms.
    https://albumentations.ai/docs/3-basic-usage/choosing-augmentations/#example-for-aerialmedical-images-with-rotational-symmetry
    """
    if is_training:
        return A.Compose(
            [
                # Basic Geometric
                A.SquareSymmetry(p=0.5),
                # Affine and Perspective
                A.Affine(
                    scale=(0.95, 1.05), translate_percent=0.1, rotate=(-45, 45), p=0.6
                ),
                # Blur
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                        A.MedianBlur(blur_limit=5, p=0.5),
                        A.MotionBlur(blur_limit=(3, 7), p=0.5),
                    ],
                    p=0.2,
                ),
                # Noise
                A.OneOf(
                    [
                        A.GaussNoise(p=0.5),
                        A.ISONoise(
                            color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5
                        ),
                        A.MultiplicativeNoise(
                            multiplier=(0.9, 1.1), per_channel=True, p=0.5
                        ),
                        A.SaltAndPepper(p=0.5),
                    ],
                    p=0.2,
                ),
                # Weather effects
                A.RandomSunFlare(p=0.2),
                A.RandomFog(p=0.2),
            ]
        )
    else:
        return None


# ====================================
# CHANNELS_LAST MEMORY FORMAT
# ====================================
def optimize_model_memory_format(model, device):
    """Convert model to channels_last for better Tensor Core utilization."""
    if torch.cuda.is_available() and device.type == "cuda":
        # Convert model to channels_last
        model = model.to(memory_format=torch.channels_last)
        logger.info("Model converted to channels_last memory format")
        
        # Verify Tensor Core support
        major, minor = torch.cuda.get_device_capability(device)
        if major >= 7:  # Volta+ architecture
            logger.info("GPU supports Tensor Cores with channels_last")
        else:
            logger.warning("GPU may not benefit from channels_last optimization")
    
    return model

def convert_input_to_channels_last(images):
    """Convert input tensors to channels_last format."""
    return images.to(memory_format=torch.channels_last)

# ====================================
# Model Creation
# ====================================
def setup_fold_specific_cache(fold):
    """Setup isolated cache for this specific fold job."""
    job_id = os.environ.get('SLURM_JOB_ID', f'local_{os.getpid()}')
    timestamp = int(time.time())
    cache_id = f"job_{job_id}_fold_{fold}_{timestamp}"
    
    base_cache = Path.home() / f".smp_cache_{cache_id}"
    hf_cache = base_cache / "huggingface"
    torch_cache = base_cache / "torch"
    
    hf_cache.mkdir(parents=True, exist_ok=True)
    torch_cache.mkdir(parents=True, exist_ok=True)
    
    original_env = {
        'HF_HOME': os.environ.get('HF_HOME'),
        'HF_HUB_CACHE': os.environ.get('HF_HUB_CACHE'),
        'TORCH_HOME': os.environ.get('TORCH_HOME'),
    }
    
    os.environ['HF_HOME'] = str(hf_cache)
    os.environ['HF_HUB_CACHE'] = str(hf_cache)
    os.environ['TORCH_HOME'] = str(torch_cache)
    
    # Keep your existing download settings
    os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'
    os.environ['REQUESTS_TIMEOUT'] = '300'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = 'true'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = 'true'
    os.environ['HF_HUB_OFFLINE'] = 'false'
    
    def cleanup():
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if base_cache.exists():
            logger.info(f"Cleaning up cache: {base_cache}")
            shutil.rmtree(base_cache, ignore_errors=True)
    
    atexit.register(cleanup)
    logger.info(f"Setup isolated cache for fold {fold}: {base_cache}")
    return cache_id, cleanup

def get_hf_cache_dir():
    """Get the current HuggingFace cache directory."""
    if 'HF_HOME' in os.environ:
        return Path(os.environ['HF_HOME'])
    elif 'HF_HUB_CACHE' in os.environ:
        return Path(os.environ['HF_HUB_CACHE'])
    else:
        return Path.home() / '.cache' / 'huggingface'

def get_torch_cache_dir():
    """Get the current PyTorch cache directory."""
    if 'TORCH_HOME' in os.environ:
        return Path(os.environ['TORCH_HOME'])
    else:
        return Path.home() / '.cache' / 'torch'

def get_lock_dir():
    """Get directory for lock files."""
    cache_dir = get_hf_cache_dir()
    return cache_dir.parent


def create_model(config):
    """
    Enhanced create_model function that handles concurrent downloads.
    """
    architecture = config["architecture"].lower()
    backbone = config["backbone"]
    
    logger.info(f"Creating {architecture} model with {backbone} backbone")
    
    # Get the actual cache location
    cache_dir = get_hf_cache_dir()
    lock_dir = get_lock_dir()
    
    logger.info(f"HF Cache directory: {cache_dir}")
    logger.info(f"Lock directory: {lock_dir}")
    
    # Create lock file in same parent directory as cache
    lock_file = lock_dir / f".{backbone.replace('/', '_')}.lock"
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    
    lock_fd = None
    
    try:
        # Acquire lock
        lock_fd = os.open(str(lock_file), os.O_CREAT | os.O_RDWR)
        
        # Wait for other downloads (max 10 minutes)
        logger.info(f"Attempting to acquire lock: {lock_file}")
        for attempt in range(120):
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                logger.info(f"Acquired lock for {backbone}")
                break
            except BlockingIOError:
                if attempt == 0:
                    logger.info(f"Waiting for {backbone} download by another job...")
                elif attempt % 12 == 0:  # Log every minute
                    logger.info(f"Still waiting for {backbone} (attempt {attempt}/120)...")
                time.sleep(5)
        else:
            raise TimeoutError(f"Timeout waiting for {backbone}")
        
        # Create model with retry logic
        return _create_model_with_retry(config, cache_dir)
        
    finally:
        if lock_fd is not None:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)
                logger.info(f"Released lock for {backbone}")
            except Exception as e:
                logger.error(f"Failed to release lock for {backbone}: {e}")


def _create_model_with_retry(config, cache_dir):
    """Model creation logic with retry for download errors."""
    architecture = config["architecture"].lower()
    backbone = config["backbone"]
    encoder_weights = config["encoder_weights"]
    
    model_creators = {
        "unet": smp.Unet,
        "unetplusplus": smp.UnetPlusPlus,
        "manet": smp.MAnet,
        "linknet": smp.Linknet,
        "fpn": smp.FPN,
        "pan": smp.PAN,
        "pspnet": smp.PSPNet,
        "segformer": smp.Segformer,
        "deeplabv3": smp.DeepLabV3Plus,
        "dpt": smp.DPT,
        "upernet": smp.UPerNet,
    }
    
    if architecture not in model_creators:
        raise ValueError(f"Architecture '{architecture}' not supported")
    
    model_class = model_creators[architecture]
    
    model_params = {
        "encoder_name": backbone,
        "encoder_weights": encoder_weights,
        "in_channels": 3,
        "classes": config["num_classes"],
        "activation": None,
    }
    
    if architecture == "dpt":
        model_params["dynamic_img_size"] = True
    
    # Retry logic for download errors
    max_attempts = 10
    for attempt in range(max_attempts):
        try:
            logger.info(f"Creating model (attempt {attempt + 1}/{max_attempts})...")
            model = model_class(**model_params)
            logger.info("Model created successfully")
            return model
            
        except Exception as e:
            error_str = str(e)
            logger.error(f"Model creation failed on attempt {attempt + 1}: {error_str}")
            # # Check for download-related errors that need cache clearing
            # download_errors = [
            #     "416",
            #     "Requested Range Not Satisfiable",
            #     "Consistency check failed",
            #     "file should be of size",
            #     "network issues while downloading"
            # ]
            
            # needs_cache_clear = any(err in error_str for err in download_errors)
            
            # if needs_cache_clear and attempt < max_attempts - 1:
            #     logger.warning(f"Download error on attempt {attempt + 1}, clearing cache and retrying...")
            #     logger.warning(f"Error details: {error_str}")
                
            #     # Clear the actual cache directory
            #     if cache_dir.exists():
            #         try:
            #             shutil.rmtree(cache_dir)
            #             logger.info(f"Cleared cache directory: {cache_dir}")
            #             # Recreate it
            #             cache_dir.mkdir(parents=True, exist_ok=True)
            #         except Exception as cache_e:
            #             logger.error(f"Failed to clear cache directory: {cache_e}")
                
            #     wait_time = 30 * (attempt + 1)
            #     logger.info(f"Waiting {wait_time} seconds before retry...")
            #     time.sleep(wait_time)
            #     continue
            
            # # Re-raise the error if it's not a download error or last attempt
            # logger.error(f"Model creation failed on attempt {attempt + 1}: {e}")
            # if attempt == max_attempts - 1:
            #     logger.error(f"Model creation failed after {max_attempts} attempts")
            # raise e


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ====================================
# Metrics Calculation
# ====================================
def calculate_metrics_smp(pred_logits, target):
    """
    Calculate all metrics using the current SMP functional API.

    Args:
        pred_logits: Raw model outputs (logits) - shape [B, H, W]
        target: Binary ground truth masks - shape [B, H, W]

    Returns:
        Dictionary with all calculated metrics
    """
    try:
        # Apply sigmoid to convert logits to probabilities
        pred_probs = torch.sigmoid(pred_logits)

        # Ensure target is binary (0 or 1) and handle potential NaN values
        target_binary = (target > 0.5).long()  # Convert to long for SMP metrics

        # Check for invalid values
        if torch.isnan(pred_probs).any() or torch.isnan(target_binary.float()).any():
            logger.warning("NaN values detected in predictions or targets")
            return {
                "iou": 0.0,
                "f1_score": 0.0,
                "accuracy": 0.0,
                "recall": 0.0,
                "precision": 0.0,
            }

        # Add channel dimension for SMP metrics (they expect [B, C, H, W])
        if len(pred_probs.shape) == 3:
            pred_probs = pred_probs.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
        if len(target_binary.shape) == 3:
            target_binary = target_binary.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]

        # Calculate all metrics using SMP functional API
        with torch.no_grad():
            # First get the statistics (tp, fp, fn, tn)
            tp, fp, fn, tn = smp.metrics.get_stats(
                pred_probs,
                target_binary,
                mode="binary",  # Use 'binary' mode for single class segmentation
                threshold=0.5,
            )

            # Log the statistics for debugging
            logger.debug(
                f"Stats - TP: {tp.sum()}, FP: {fp.sum()}, FN: {fn.sum()}, TN: {tn.sum()}"
            )

            # Then compute metrics with the statistics
            metrics = {
                "iou": smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").item(),
                "f1_score": smp.metrics.f1_score(
                    tp, fp, fn, tn, reduction="micro"
                ).item(),
                "accuracy": smp.metrics.accuracy(
                    tp, fp, fn, tn, reduction="micro"
                ).item(),
                "recall": smp.metrics.recall(tp, fp, fn, tn, reduction="micro").item(),
                "precision": smp.metrics.precision(
                    tp, fp, fn, tn, reduction="micro"
                ).item(),
            }

            # Log the calculated metrics
            logger.debug(f"Calculated metrics: {metrics}")

            # Handle potential NaN or inf values with better logic
            for metric_name, metric_value in metrics.items():
                if not torch.isfinite(torch.tensor(metric_value)):
                    if metric_name in ["precision", "recall"]:
                        # For precision/recall, NaN usually means no positive predictions/targets
                        # Set to 0.0 for these edge cases
                        logger.debug(
                            f"Invalid {metric_name}: {metric_value}, setting to 0.0"
                        )
                        metrics[metric_name] = 0.0
                    elif metric_name == "f1_score" and not torch.isfinite(
                        torch.tensor(metric_value)
                    ):
                        # F1 depends on precision/recall, so if either is NaN, F1 will be NaN
                        metrics[metric_name] = 0.0
                    else:
                        metrics[metric_name] = 0.0
                    logger.debug(
                        f"Replaced invalid {metric_name} value ({metric_value}) with 0.0"
                    )

        return metrics

    except Exception as e:
        logger.error(f"Error in metrics calculation: {e}")
        return {
            "iou": 0.0,
            "f1_score": 0.0,
            "accuracy": 0.0,
            "recall": 0.0,
            "precision": 0.0,
        }


# ====================================
# Training Functions
# ====================================
def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, accumulation_steps=1):
    """Train model for one epoch with mixed precision."""
    model.train()
    total_loss = 0
    all_metrics = {
        "iou": [], "f1_score": [], "accuracy": [], "recall": [], "precision": [],
    }
    total_samples = 0
    consecutive_errors = 0
    max_grad_norm = 1.0
    grad_norm = 0.0
    
    # total_memory = torch.cuda.get_device_properties(device).total_memory
    # memory_warning_threshold = total_memory * 0.90
    # memory_critical_threshold = total_memory * 0.95

    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    
    # Zero gradients
    optimizer.zero_grad(set_to_none=True)

    try:
        for batch_idx, (images, masks) in enumerate(progress_bar):
            try:
                # # Check memory usage
                # if batch_idx % 20 == 0 and batch_idx > 0:
                #     current_memory = torch.cuda.memory_reserved(device) 
                #     if current_memory > memory_warning_threshold:
                #         logger.warning(f"High memory usage before batch {batch_idx}: {current_memory / 1024**3:.1f}GB")
                #         clear_memory()
                #         current_memory = torch.cuda.memory_reserved(device) 
                        
                #         if current_memory > memory_critical_threshold:
                #             logger.error(f"Critical memory usage: {current_memory / 1024**3:.1f}GB - skipping batch")
                #             continue

                # Convert to channels_last
                images = images.to(memory_format=torch.channels_last)
                
                # Move data to device
                if images.device != device:
                    images = images.to(device, non_blocking=True)
                if masks.device != device:
                    masks = masks.to(device, non_blocking=True)
                
                batch_size = images.size(0)
                total_samples += batch_size
                
                # Input validation
                if torch.isnan(images).any() or torch.isnan(masks).any():
                    logger.error(f"NaN in input data at batch {batch_idx}! Skipping...")
                    consecutive_errors += 1
                    if consecutive_errors >= 3:
                        logger.error("Too many consecutive input errors, stopping training")
                        break
                    continue
                
                # Mixed precision with gradient scaling monitoring
                with autocast(device_type="cuda"):
                    outputs = model(images)
                    
                    # NaN detection in outputs
                    if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                        logger.error(f"NaN/Inf in model outputs at batch {batch_idx}!")
                        consecutive_errors += 1
                        if consecutive_errors >= 3:
                            logger.error("Too many consecutive NaN outputs, stopping training")
                            break
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    
                    loss = criterion(outputs.squeeze(1), masks)

                    # Scale loss by accumulation steps
                    loss = loss / accumulation_steps                
                
                # Loss validation
                if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 100.0:
                    logger.error(f"Invalid loss at batch {batch_idx}: {loss.item()}")
                    consecutive_errors += 1
                    if consecutive_errors >= 3:
                        logger.error("Too many consecutive loss errors, stopping training")
                        break
                    optimizer.zero_grad(set_to_none=True)
                    continue
                
                # Check memory before backward pass
                # if batch_idx % 20 == 0 and batch_idx > 0:
                #     current_memory = torch.cuda.memory_reserved(device)
                #     if current_memory > memory_critical_threshold:
                #         logger.warning(f"High memory before backward pass: {current_memory / 1024**3:.1f}GB - clearing cache")
                #         clear_memory()

                # Mixed precision backward pass
                scaler.scale(loss).backward()
                
                # Only update weights every accumulation_steps
                if (batch_idx + 1) % accumulation_steps == 0:
                    # Unscale before checking/clipping gradients
                    scaler.unscale_(optimizer)
                    
                    # Clip gradients
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                    
                    # Update weights
                    scaler.step(optimizer)
                    scaler.update()
                    
                    # Zero gradients for next accumulation cycle
                    optimizer.zero_grad(set_to_none=True)
                    
                    if torch.isfinite(grad_norm):
                        consecutive_errors = 0
                    else:
                        logger.error(f"Non-finite gradients at batch {batch_idx}, grad_norm: {grad_norm}")
                        consecutive_errors += 1
                        if consecutive_errors >= 3:
                            logger.error("Too many gradient errors, stopping training")
                            break
               
                # Accumulate loss
                total_loss += loss.item() * accumulation_steps
                
                # Calculate metrics
                with torch.no_grad(), autocast(device_type="cuda"):
                    batch_metrics = calculate_metrics_smp(outputs.squeeze(1), masks)
                    for key, value in batch_metrics.items():
                        all_metrics[key].append(value)
                
                # Update progress bar
                # _, mem_reserved_pct, mem_max_allocated_pct = get_memory_usage(device)

                if (batch_idx + 1) % accumulation_steps == 0:
                    # Show gradient norm when computed
                    progress_bar.set_postfix({
                        "loss": f"{loss.item() * accumulation_steps:.4f}",
                        "iou": f"{batch_metrics['iou']:.4f}",
                        "grad": f"{grad_norm:.2f}",
                    })
                else:
                    # Don't show gradient norm during accumulation
                    progress_bar.set_postfix({
                        "loss": f"{loss.item() * accumulation_steps:.4f}",
                        "iou": f"{batch_metrics['iou']:.4f}",
                    })
                
                # Clear intermediate variables
                del images, masks, outputs, loss
                
                # Periodic memory cleanup
                if batch_idx % 50 == 0 and batch_idx > 0:
                    clear_memory()
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error(f"GPU OOM at batch {batch_idx}. Current memory: {torch.cuda.memory_allocated(device) / 1024**3:.1f}GB")
                    clear_memory()
                    optimizer.zero_grad(set_to_none=True)
                    consecutive_errors += 1

                    if consecutive_errors >= 3:
                        logger.error("Multiple OOM errors - consider reducing accumulation_steps or batch_size")
                        logger.error("Current memory stats:")
                        logger.error(f"  Allocated: {torch.cuda.memory_allocated(device) / 1024**3:.1f}GB")
                        logger.error(f"  Cached: {torch.cuda.memory_reserved(device) / 1024**3:.1f}GB")
                        break
                    continue
                else:
                    raise e
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                optimizer.zero_grad(set_to_none=True)
                consecutive_errors += 1
                if consecutive_errors >= 3:
                    break
                continue
                
    finally:
        mem_info, _, _ = get_memory_usage(device)
        logger.info(f"COMPLETED epoch - {mem_info}")
        # Final cleanup
        clear_memory()
        
    # Calculate averages
    avg_metrics = {}
    for key, values in all_metrics.items():
        if values:
            avg_metrics[key] = np.mean(values)
        else:
            logger.warning(f"No valid {key} values collected")
            avg_metrics[key] = 0.0
    
    avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0
    
    logger.info(f"Training epoch completed: {total_samples} samples processed")
    if consecutive_errors > 0:
        logger.warning(f"Had {consecutive_errors} consecutive errors during training")
    
    return avg_loss, avg_metrics


def validate_one_epoch(model, val_loader, criterion, device):
    """Validate model for one epoch with mixed precision."""
    model.eval()
    total_loss = 0
    all_metrics = {
        "iou": [],
        "f1_score": [],
        "accuracy": [],
        "recall": [],
        "precision": [],
    }
    total_samples = 0

    progress_bar = tqdm(val_loader, desc="Validation", leave=False)

    try:
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(progress_bar):
                try:
                    # Convert to channels_last
                    images = images.to(memory_format=torch.channels_last)

                    # Move data to device efficiently
                    if images.device != device:
                        images = images.to(device, non_blocking=True)
                    if masks.device != device:
                        masks = masks.to(device, non_blocking=True)

                    batch_size = images.size(0)
                    total_samples += batch_size

                    # Mixed precision forward pass
                    with autocast(device_type="cuda"):
                        outputs = model(images)
                        loss = criterion(outputs.squeeze(1), masks)

                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.error(
                            f"VALIDATION ERROR: NaN/Inf loss at batch {batch_idx}!"
                        )
                        logger.error(
                            f"  Outputs stats: min={outputs.min():.4f}, max={outputs.max():.4f}, mean={outputs.mean():.4f}"
                        )
                        logger.error(
                            f"  Masks stats: min={masks.min():.4f}, max={masks.max():.4f}, mean={masks.mean():.4f}"
                        )
                        continue

                    total_loss += loss.item()

                    # SMP metrics calculation (GPU accelerated)
                    with autocast(device_type="cuda"):
                        batch_metrics = calculate_metrics_smp(outputs.squeeze(1), masks)
                        for key, value in batch_metrics.items():
                            all_metrics[key].append(value)

                    # Update progress bar
                    progress_bar.set_postfix(
                        {
                            "loss": f"{loss.item():.4f}",
                            "iou": f"{batch_metrics['iou']:.4f}",
                        }
                    )

                    # Clear variables
                    del images, masks, outputs, loss

                    # Periodic memory cleanup
                    if batch_idx % 25 == 0 and batch_idx > 0:
                        clear_memory()

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.error(f"GPU OOM during validation at batch {batch_idx}")
                        clear_memory()
                        continue
                    else:
                        raise e
                except Exception as e:
                    logger.error(f"Error in validation batch {batch_idx}: {e}")
                    continue

    finally:
        # Final cleanup
        clear_memory()

    # Calculate averages - handle empty metrics gracefully
    avg_metrics = {}
    for key, values in all_metrics.items():
        if values:
            avg_metrics[key] = np.mean(values)
        else:
            logger.warning(f"No valid validation {key} values collected")
            avg_metrics[key] = 0.0

    avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0

    logger.info(f"Validation epoch completed: {total_samples} samples processed")

    return avg_loss, avg_metrics


def calculate_iou_simple(pred_binary, target_binary):
    """Simple IoU calculation for visualization purposes."""
    with torch.no_grad():
        # Ensure binary masks
        pred = (pred_binary > 0.5).float()
        target = (target_binary > 0.5).float()

        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection

        if union == 0:
            return 1.0 if pred.sum() == 0 and target.sum() == 0 else 0.0
        else:
            return (intersection / union).item()


# ====================================
# Test Evaluation Function
# ====================================
def evaluate_on_test_set_smp(model, test_loader, device, save_dir):
    """Test evaluation using SMP metrics with visualization"""
    evaluation_start_time = time.time()
    logger.info("Evaluating on test set with SMP metrics...")

    model.eval()
    all_metrics = {
        "iou": [],
        "f1_score": [],
        "accuracy": [],
        "recall": [],
        "precision": [],
    }
    per_image_results = []

    # For visualization - collect some samples
    sample_data = []

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(test_loader, desc="Testing")):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with autocast(device_type="cuda"):
                outputs = model(images)
                # Keep raw logits for metrics (no sigmoid here)
                raw_outputs = outputs.squeeze(1)

            # Calculate metrics for each image in batch
            for i in range(len(images)):
                # Pass raw logits to metrics (calculate_metrics_smp will apply sigmoid internally)
                single_output = raw_outputs[i : i + 1]
                single_mask = masks[i : i + 1]

                # This will apply sigmoid ONCE inside calculate_metrics_smp
                batch_metrics = calculate_metrics_smp(single_output, single_mask)

                for key, value in batch_metrics.items():
                    all_metrics[key].append(value)

                per_image_results.append(
                    {"batch": batch_idx, "image_idx": i, **batch_metrics}
                )

                # Collect samples for visualization (diversified selection)
                if len(sample_data) < 12:
                    # Apply sigmoid ONLY for visualization
                    viz_prediction = torch.sigmoid(single_output)
                    viz_prediction_binary = (viz_prediction > 0.5).float()

                    sample_data.append(
                        {
                            "image": images[i].cpu(),
                            "mask": masks[i].cpu(),
                            "prediction": viz_prediction_binary.cpu().squeeze(0),
                            "iou": batch_metrics["iou"],
                        }
                    )

            # Clean up variables
            del images, masks, outputs, raw_outputs

            if batch_idx % 10 == 0:
                clear_memory()

    # Calculate average metrics
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}

    # Log results
    logger.info("Test Set Results (SMP metrics):")
    for metric, value in avg_metrics.items():
        logger.info(f"  {metric.upper()}: {value:.4f}")

    evaluation_time = time.time() - evaluation_start_time

    # Save detailed results
    results_file = os.path.join(save_dir, "test_results_smp.json")
    with open(results_file, "w") as f:
        json.dump(
            {
                "average_metrics": avg_metrics,
                "per_image_results": per_image_results,
                # NEW: Add evaluation timing
                "evaluation_timing": {
                    "evaluation_seconds": evaluation_time,
                    "evaluation_time_formatted": f"{int(evaluation_time // 60):02d}:{int(evaluation_time % 60):02d}",
                    "total_test_samples": len(per_image_results),
                    "average_seconds_per_sample": evaluation_time / len(per_image_results) if per_image_results else 0,
                    "evaluation_timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                }
            },
            f,
            indent=4,
        )

    # Save CSV results
    csv_file = os.path.join(save_dir, "test_results.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "batch",
                "image_idx",
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "iou",
            ],
        )
        writer.writeheader()
        writer.writerows(per_image_results)

    # Create test visualizations
    visualize_test_results(sample_data, save_dir)

    return avg_metrics


def visualize_test_results(sample_data, save_dir, samples_per_row=3):
    num_samples = len(sample_data)
    num_rows = (num_samples + samples_per_row - 1) // samples_per_row

    # Sort samples by IoU for better visualization
    sample_data_sorted = sorted(sample_data, key=lambda x: x["iou"], reverse=True)

    fig, axes = plt.subplots(num_rows, samples_per_row * 3, figsize=(8, 6 * num_rows))
    fig.suptitle(
        "Test Set Predictions (Sorted by IoU - Best to Worst)",
        fontsize=18,
        fontweight="bold",
    )

    # Handle single row case
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    elif num_samples == 1:
        axes = axes.reshape(num_rows, -1)

    # Calculate statistics
    ious = [s["iou"] for s in sample_data]
    mean_iou = np.mean(ious)
    std_iou = np.std(ious)

    for i, sample in enumerate(sample_data_sorted):
        if i >= num_samples:
            break

        row = i // samples_per_row
        col_start = (i % samples_per_row) * 3

        # Color coding based on IoU performance
        iou_val = sample["iou"]
        if iou_val >= 0.7:
            border_color = "#2e7d32"  # Green for good
            performance = "Excellent"
        elif iou_val >= 0.5:
            border_color = "#fbc02d"  # Yellow for okay
            performance = "Good"
        elif iou_val >= 0.3:
            border_color = "#ff9800"  # Orange for poor
            performance = "Fair"
        else:
            border_color = "#d32f2f"  # Red for very poor
            performance = "Poor"

        # Original image
        img = sample["image"].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        axes[row, col_start].imshow(img)
        axes[row, col_start].set_title(
            f"Input #{i + 1}", fontsize=11, fontweight="bold"
        )
        axes[row, col_start].axis("off")

        # Colored border
        for spine in axes[row, col_start].spines.values():
            spine.set_visible(True)
            spine.set_linewidth(3)
            spine.set_color(border_color)

        # Ground truth
        axes[row, col_start + 1].imshow(
            sample["mask"].numpy(), cmap="gray", vmin=0, vmax=1
        )
        axes[row, col_start + 1].set_title(
            "Ground Truth", fontsize=11, fontweight="bold"
        )
        axes[row, col_start + 1].axis("off")

        for spine in axes[row, col_start + 1].spines.values():
            spine.set_visible(True)
            spine.set_linewidth(3)
            spine.set_color(border_color)

        # Prediction
        axes[row, col_start + 2].imshow(
            sample["prediction"].numpy(), cmap="gray", vmin=0, vmax=1
        )
        axes[row, col_start + 2].set_title(
            f"{performance}\nIoU: {iou_val:.3f}",
            fontsize=11,
            fontweight="bold",
            color=border_color,
        )
        axes[row, col_start + 2].axis("off")

        for spine in axes[row, col_start + 2].spines.values():
            spine.set_visible(True)
            spine.set_linewidth(3)
            spine.set_color(border_color)

    # Hide empty subplots
    total_subplots = num_rows * samples_per_row * 3
    for i in range(num_samples * 3, total_subplots):
        row = i // (samples_per_row * 3)
        col = i % (samples_per_row * 3)
        if row < axes.shape[0] and col < axes.shape[1]:
            axes[row, col].axis("off")

    # Statistics text box
    stats_text = (
        f"Test Statistics:\n"
        f"Mean IoU: {mean_iou:.4f} ± {std_iou:.4f}\n"
        f"Best IoU: {max(ious):.4f}\n"
        f"Worst IoU: {min(ious):.4f}\n"
        f"Samples > 0.5 IoU: {sum(1 for iou in ious if iou > 0.5)}/{len(ious)}"
    )

    fig.text(
        0.02,
        0.02,
        stats_text,
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
        verticalalignment="bottom",
    )

    plt.tight_layout()
    save_path = os.path.join(save_dir, "test_predictions.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Test visualizations saved to: {save_path}")
    return save_path


# ====================================
# Visualization Functions
# ====================================
def plot_training_history(history, fold, save_dir):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Training History - Fold {fold}", fontsize=16)

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss plot
    axes[0, 0].plot(
        epochs, history["train_loss"], "b-", label="Training Loss", linewidth=2
    )
    axes[0, 0].plot(
        epochs, history["val_loss"], "r-", label="Validation Loss", linewidth=2
    )
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epochs")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # IoU plot
    axes[0, 1].plot(
        epochs, history["train_iou"], "b-", label="Training IoU", linewidth=2
    )
    axes[0, 1].plot(
        epochs, history["val_iou"], "r-", label="Validation IoU", linewidth=2
    )
    axes[0, 1].set_title("IoU Score")
    axes[0, 1].set_xlabel("Epochs")
    axes[0, 1].set_ylabel("IoU")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Accuracy plot
    axes[1, 0].plot(
        epochs, history["train_accuracy"], "b-", label="Training Accuracy", linewidth=2
    )
    axes[1, 0].plot(
        epochs, history["val_accuracy"], "r-", label="Validation Accuracy", linewidth=2
    )
    axes[1, 0].set_title("Accuracy")
    axes[1, 0].set_xlabel("Epochs")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # F1 Score plot
    axes[1, 1].plot(
        epochs, history["train_f1_score"], "b-", label="Training F1", linewidth=2
    )
    axes[1, 1].plot(
        epochs, history["val_f1_score"], "r-", label="Validation F1", linewidth=2
    )
    axes[1, 1].set_title("F1 Score")
    axes[1, 1].set_xlabel("Epochs")
    axes[1, 1].set_ylabel("F1 Score")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"training_history_fold_{fold}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Training history plot saved: {save_path}")
    return save_path


def visualize_predictions(model, val_loader, device, fold, save_dir, num_samples=3):
    """Visualize model predictions"""
    logger.info("Creating validation predictions visualization...")

    model.eval()

    # Get a batch of data
    try:
        images, masks = next(iter(val_loader))
        images = images.to(device)
        masks = masks.to(device)

        with torch.no_grad():
            outputs = model(images)

            # Keep raw outputs for metrics, apply sigmoid only for visualization
            raw_outputs = outputs.squeeze(1)

            # Apply sigmoid for visualization purposes only
            predictions_for_viz = torch.sigmoid(raw_outputs)
            predictions_binary = (predictions_for_viz > 0.5).float()

        # Plot samples
        fig, axes = plt.subplots(num_samples, 3, figsize=(4.5, 2 * num_samples))
        fig.suptitle(f"Validation Predictions - Fold {fold}", fontsize=12)

        # Handle single sample case
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(min(num_samples, len(images))):
            # Original image
            img = images[i].cpu().permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)  # Ensure valid range
            axes[i, 0].imshow(img)
            if i == 0:
                axes[i, 0].set_title("Input Image", fontsize=8, fontweight="bold")
            axes[i, 0].axis("off")

            # Ground truth
            gt = masks[i].cpu().numpy()
            axes[i, 1].imshow(gt, cmap="gray", vmin=0, vmax=1)
            if i == 0:
                axes[i, 1].set_title("Ground Truth", fontsize=8, fontweight="bold")
            axes[i, 1].axis("off")

            # Prediction with IoU calculation
            pred = predictions_binary[i].cpu().numpy()

            # FIXED: Calculate IoU using raw logits (no double sigmoid)
            single_pred = raw_outputs[i : i + 1]  # Raw logits with batch dimension
            single_mask = masks[i : i + 1]

            with torch.no_grad():
                # calculate_metrics_smp will apply sigmoid internally (only once)
                metrics = calculate_metrics_smp(single_pred, single_mask)
                iou_score = metrics["iou"]

            axes[i, 2].imshow(pred, cmap="gray", vmin=0, vmax=1)
            if i == 0:
                axes[i, 2].set_title("Prediction", fontsize=8, fontweight="bold")

            # IoU text below each prediction (for all samples, not just first)
            axes[i, 2].text(
                0.5,
                -0.05,
                f"IoU: {iou_score:.3f}",
                transform=axes[i, 2].transAxes,
                ha="center",
                va="top",
                fontsize=10,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )
            axes[i, 2].axis("off")

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"val_predictions_fold_{fold}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Validation predictions saved: {save_path}")
        return save_path

    except Exception as e:
        logger.error(f"Failed to create validation predictions: {e}")
        return None


def save_epoch_graphics(history, epoch, fold, graphics_dir, val_loader, model, device):
    """Save training graphics for each epoch in separate folder"""
    # Create epoch-specific directory
    epoch_dir = os.path.join(graphics_dir, f"epoch_{epoch:03d}")
    os.makedirs(epoch_dir, exist_ok=True)

    if len(history["train_loss"]) < 1:
        return

    # 1. Training Progress Plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Training Progress - Fold {fold} - Epoch {epoch}", fontsize=16)

    epochs_range = range(1, len(history["train_loss"]) + 1)

    # Loss plot
    axes[0, 0].plot(
        epochs_range, history["train_loss"], "b-", label="Training Loss", linewidth=2
    )
    axes[0, 0].plot(
        epochs_range, history["val_loss"], "r-", label="Validation Loss", linewidth=2
    )
    axes[0, 0].set_title("Loss Over Time")
    axes[0, 0].set_xlabel("Epochs")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # IoU plot
    axes[0, 1].plot(
        epochs_range, history["train_iou"], "b-", label="Training IoU", linewidth=2
    )
    axes[0, 1].plot(
        epochs_range, history["val_iou"], "r-", label="Validation IoU", linewidth=2
    )
    axes[0, 1].set_title("IoU Score Over Time")
    axes[0, 1].set_xlabel("Epochs")
    axes[0, 1].set_ylabel("IoU")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Accuracy plot
    axes[1, 0].plot(
        epochs_range,
        history["train_accuracy"],
        "b-",
        label="Training Accuracy",
        linewidth=2,
    )
    axes[1, 0].plot(
        epochs_range,
        history["val_accuracy"],
        "r-",
        label="Validation Accuracy",
        linewidth=2,
    )
    axes[1, 0].set_title("Accuracy Over Time")
    axes[1, 0].set_xlabel("Epochs")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # F1 Score plot
    axes[1, 1].plot(
        epochs_range, history["train_f1_score"], "b-", label="Training F1", linewidth=2
    )
    axes[1, 1].plot(
        epochs_range, history["val_f1_score"], "r-", label="Validation F1", linewidth=2
    )
    axes[1, 1].set_title("F1 Score Over Time")
    axes[1, 1].set_xlabel("Epochs")
    axes[1, 1].set_ylabel("F1 Score")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    progress_path = os.path.join(epoch_dir, f"training_progress_epoch_{epoch:03d}.png")
    plt.savefig(progress_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Current Metrics Summary Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    if len(history["train_loss"]) > 0:
        current_metrics = {
            "Train Loss": history["train_loss"][-1],
            "Val Loss": history["val_loss"][-1],
            "Train IoU": history["train_iou"][-1],
            "Val IoU": history["val_iou"][-1],
            "Train Acc": history["train_accuracy"][-1],
            "Val Acc": history["val_accuracy"][-1],
            "Train F1": history["train_f1_score"][-1],
            "Val F1": history["val_f1_score"][-1],
        }

        # Split into train/val for better visualization
        train_metrics = {
            k.replace("Train ", ""): v
            for k, v in current_metrics.items()
            if "Train" in k
        }
        val_metrics = {
            k.replace("Val ", ""): v for k, v in current_metrics.items() if "Val" in k
        }

        x = np.arange(len(train_metrics))
        width = 0.35

        bars1 = ax.bar(
            x - width / 2,
            list(train_metrics.values()),
            width,
            label="Train",
            alpha=0.8,
            color="#1f77b4",
        )
        bars2 = ax.bar(
            x + width / 2,
            list(val_metrics.values()),
            width,
            label="Validation",
            alpha=0.8,
            color="#ff7f0e",
        )

        ax.set_xlabel("Metrics")
        ax.set_ylabel("Values")
        ax.set_title(f"Current Metrics - Epoch {epoch}")
        ax.set_xticks(x)
        ax.set_xticklabels(list(train_metrics.keys()))
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    metrics_path = os.path.join(epoch_dir, f"current_metrics_epoch_{epoch:03d}.png")
    plt.savefig(metrics_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Sample Predictions with Best/Worst (every 5 epochs to save space)
    try:
        model.eval()

        # Collect ALL validation samples with their IoU scores
        all_samples = []

        with torch.no_grad():
            for batch_images, batch_masks in val_loader:
                batch_images = batch_images.to(device)
                batch_masks = batch_masks.to(device)

                batch_outputs = model(batch_images)
                batch_predictions = torch.sigmoid(batch_outputs.squeeze(1))
                batch_predictions_binary = (batch_predictions > 0.5).float()

                # Calculate IoU for each sample in the batch
                for i in range(len(batch_images)):
                    iou = calculate_iou_simple(
                        batch_predictions_binary[i : i + 1], batch_masks[i : i + 1]
                    )
                    all_samples.append(
                        {
                            "image": batch_images[i].cpu(),
                            "mask": batch_masks[i].cpu(),
                            "prediction": batch_predictions_binary[i].cpu(),
                            "iou": iou,
                        }
                    )
                # Clean up
                del (
                    batch_images,
                    batch_masks,
                    batch_outputs,
                    batch_predictions,
                    batch_predictions_binary,
                )
                clear_memory()

        # Sort all samples by IoU score (ascending order)
        all_samples.sort(key=lambda x: x["iou"])

        # Get exactly 5 WORST predictions (lowest IoU scores)
        worst_samples = all_samples[:5]  # First 5 after sorting = worst

        # Get exactly 5 BEST predictions (highest IoU scores)
        best_samples = all_samples[-5:]  # Last 5 after sorting = best
        best_samples.reverse()  # Reverse to show best → good order in visualization

        # Verify the selection worked correctly
        worst_ious = [s["iou"] for s in worst_samples]
        best_ious = [s["iou"] for s in best_samples]

        logger.info(f"  Worst 5 IoU scores: {[f'{iou:.4f}' for iou in worst_ious]}")
        logger.info(f"  Best 5 IoU scores: {[f'{iou:.4f}' for iou in best_ious]}")

        # Create WORST predictions visualization
        create_prediction_visualization(
            samples=worst_samples,
            title="Worst Predictions",
            subtitle=f"5 Lowest IoU Scores - Fold {fold}, Epoch {epoch}",
            save_path=os.path.join(
                epoch_dir, f"worst_predictions_epoch_{epoch:03d}.png"
            ),
            color_theme="red",
        )

        # Create BEST predictions visualization
        create_prediction_visualization(
            samples=best_samples,
            title="Best Predictions",
            subtitle=f"5 Highest IoU Scores - Fold {fold}, Epoch {epoch}",
            save_path=os.path.join(
                epoch_dir, f"best_predictions_epoch_{epoch:03d}.png"
            ),
            color_theme="green",
        )

        # Log comprehensive statistics
        if all_samples:
            ious = [s["iou"] for s in all_samples]
            logger.info(f"  Validation IoU Statistics - Epoch {epoch}:")
            logger.info(f"    Total samples: {len(all_samples)}")
            logger.info(f"    IoU range: {min(ious):.4f} - {max(ious):.4f}")
            logger.info(f"    Mean: {np.mean(ious):.4f}, Std: {np.std(ious):.4f}")
            logger.info(f"    Median: {np.median(ious):.4f}")
            logger.info(
                f"    Samples with IoU > 0.5: {sum(1 for iou in ious if iou > 0.5)}/{len(ious)}"
            )
            logger.info(
                f"    Samples with IoU > 0.1: {sum(1 for iou in ious if iou > 0.1)}/{len(ious)}"
            )

    except Exception as e:
        logger.warning(f"Could not create best/worst predictions plot: {e}")

    logger.info(f"  Graphics saved to: {epoch_dir}")
    return epoch_dir


def create_prediction_visualization(
    samples, title, subtitle, save_path, color_theme="blue"
):
    """Version with IoU text below each prediction image."""

    # Set color scheme based on theme
    if color_theme == "red":
        title_color = "#d32f2f"
    elif color_theme == "green":
        title_color = "#2e7d32"
    else:
        title_color = "#1976d2"

    # Create figure with vertical space
    fig = plt.figure(figsize=(8, 15))  # Even more height
    fig.patch.set_facecolor("white")

    # Main title
    fig.suptitle(title, fontsize=20, fontweight="bold", color=title_color, y=0.96)

    # Subtitle
    if subtitle:
        fig.text(
            0.5,
            0.92,
            subtitle,
            fontsize=14,
            fontweight="normal",
            color="#555555",
            ha="center",
        )

    # Grid with generous vertical spacing
    gs = fig.add_gridspec(
        5,
        3,
        hspace=0.25,  # Even more vertical space
        wspace=0.005,  # Tight horizontal
        left=0.02,
        right=0.98,
        top=0.88,
        bottom=0.02,
    )

    for i, sample in enumerate(samples):
        row = i

        # Input Image
        ax1 = fig.add_subplot(gs[row, 0])
        img = sample["image"].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        ax1.imshow(img)

        if i == 0:
            ax1.set_title("Input Image", fontsize=12, fontweight="bold", pad=12)
        ax1.axis("off")

        # Ground Truth Mask
        ax2 = fig.add_subplot(gs[row, 1])
        gt = sample["mask"].numpy()
        ax2.imshow(gt, cmap="gray", vmin=0, vmax=1)

        if i == 0:
            ax2.set_title("Ground Truth", fontsize=12, fontweight="bold", pad=12)
        ax2.axis("off")

        # Prediction
        ax3 = fig.add_subplot(gs[row, 2])
        pred = sample["prediction"].numpy()
        ax3.imshow(pred, cmap="gray", vmin=0, vmax=1)

        # Color-coded IoU value
        iou_val = sample["iou"]
        if iou_val < 0.1:
            iou_color = "#d32f2f"
        elif iou_val < 0.3:
            iou_color = "#ff9800"
        elif iou_val < 0.6:
            iou_color = "#fbc02d"
        else:
            iou_color = "#2e7d32"

        if i == 0:
            ax3.set_title("Prediction", fontsize=12, fontweight="bold", pad=12)

        ax3.axis("off")

        # IoU text BELOW the image (positive y value beyond image area)
        ax3.text(
            0.5,
            -0.05,
            f"IoU: {iou_val:.3f}",
            transform=ax3.transAxes,
            verticalalignment="top",
            horizontalalignment="center",
            fontsize=11,
            fontweight="bold",
            color=iou_color,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                alpha=0.95,
                edgecolor=iou_color,
                linewidth=1.5,
            ),
        )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        pad_inches=0.15,
    )
    plt.close()

    logger.info(f"  Saved {title.lower()} visualization: {save_path}")


def create_augmentation_report(train_dataset, graphics_dir, num_samples=2):
    """Create visualizations augmentation for the report"""
    logger.info("Creating augmentation report...")

    # All augmentations organized by category
    full_pipeline = A.Compose(
            [
                A.SquareSymmetry(p=0.5),
                A.Affine(
                    scale=(0.95, 1.05), translate_percent=0.1, rotate=(-45, 45), p=0.6
                ),
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                        A.MedianBlur(blur_limit=5, p=0.5),
                        A.MotionBlur(blur_limit=(3, 7), p=0.5),
                    ],
                    p=0.2,
                ),
                A.OneOf(
                    [
                        A.GaussNoise(p=0.5),
                        A.ISONoise(
                            color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5
                        ),
                        A.MultiplicativeNoise(
                            multiplier=(0.9, 1.1), per_channel=True, p=0.5
                        ),
                        A.SaltAndPepper(p=0.5),
                    ],
                    p=0.2,
                ),
                A.RandomSunFlare(p=0.2),
                A.RandomFog(p=0.2),
            ]
        )
    categories = {
        "Symétrie": [
            ("SquareSymmetry 1", A.Compose([A.SquareSymmetry(p=1.0)])),
            ("SquareSymmetry 2", A.Compose([A.SquareSymmetry(p=1.0)])),
        ],
        "Echelle": [
            ("Affine scale ±5% 1", A.Compose([A.Affine(scale=(0.95, 1.05), p=1.0)])),
            ("Affine scale ±5% 2", A.Compose([A.Affine(scale=(0.95, 1.05), p=1.0)])),
        ],
        "Translation": [
            ("Affine translate 10% 1", A.Compose([A.Affine(translate_percent=0.1, p=1.0)])),
            ("Affine translate 10% 2", A.Compose([A.Affine(translate_percent=0.1, p=1.0)])),
        ],
        "Rotation": [
            ("Affine rotate ±45° 1", A.Compose([A.Affine(rotate=(-45, 45), p=1.0)])),
            ("Affine rotate ±45° 2", A.Compose([A.Affine(rotate=(-45, 45), p=1.0)])),
        ],
        "Flou Gaussien": [
            ("GaussianBlur 1", A.Compose([A.GaussianBlur(blur_limit=(3, 7), p=1.0)])),
            ("GaussianBlur 2", A.Compose([A.GaussianBlur(blur_limit=(3, 7), p=1.0)])),
        ],
        "Flou Median": [
            ("MedianBlur 1", A.Compose([A.MedianBlur(blur_limit=5, p=1.0)])),
            ("MedianBlur 2", A.Compose([A.MedianBlur(blur_limit=5, p=1.0)])),
        ],
        "Flou Motion": [
            ("MotionBlur 1", A.Compose([A.MotionBlur(blur_limit=(3, 7), p=1.0)])),
            ("MotionBlur 2", A.Compose([A.MotionBlur(blur_limit=(3, 7), p=1.0)])),
        ],
        "Bruit Gaussien": [
            ("GaussNoise 1", A.Compose([A.GaussNoise(p=1.0)])),
            ("GaussNoise 2", A.Compose([A.GaussNoise(p=1.0)])),
        ],
        "Bruit ISO": [
            ("ISONoise 1",
                A.Compose([A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0)])
            ),
            ("ISONoise 2",
                A.Compose([A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0)])
            ),
        ],
        "Bruit Multiplicatif": [
            ("MultiplicativeNoise 1",
                A.Compose([A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True, p=0.5)])
            ),
            ("MultiplicativeNoise 2",
                A.Compose([A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True, p=0.5)])
            ),
        ],
        "Bruit poivre et sel": [
            ("SaltAndPepper 1", A.Compose([A.SaltAndPepper(p=1.0)])),
            ("SaltAndPepper 2", A.Compose([A.SaltAndPepper(p=1.0)])),
        ],
        "Effets météo éblouissement": [
            ("RandomSunFlare 1", A.Compose([A.RandomSunFlare(p=1.0)])),
            ("RandomSunFlare 2", A.Compose([A.RandomSunFlare(p=1.0)])),
        ],
        "Effets météo brouillard": [
            ("RandomFog 1", A.Compose([A.RandomFog(p=1.0)])),
            ("RandomFog 2", A.Compose([A.RandomFog(p=1.0)])),
        ],
        "Exemples complets 1": [
            ("Full Pipeline 1", full_pipeline),
            ("Full Pipeline 2", full_pipeline),
        ],
        "Exemples complets 2": [
            ("Full Pipeline 3", full_pipeline),
            ("Full Pipeline 4", full_pipeline),
        ],
    }

    for sample_idx in range(num_samples):
        # Get original sample
        original_transform = train_dataset.transform
        train_dataset.transform = None
        orig_image, orig_mask = train_dataset[sample_idx]
        train_dataset.transform = original_transform

        orig_image_np = (orig_image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        orig_mask_np = orig_mask.numpy().astype(np.uint8)

        # Create one page per category
        for category_name, augmentations in categories.items():
            fig = plt.figure(figsize=(6.5, 8))  # Keep same size
            fig.patch.set_facecolor("white")

            # Title
            fig.suptitle(
                f"{category_name}",
                fontsize=12,
                fontweight="bold",
                y=0.91,
            )

            # Grid: 3 rows (image/mask/overlay) x (1 original + n augmentations) columns
            n_augs = len(augmentations)
            total_cols = n_augs + 1  # +1 for original
            gs = fig.add_gridspec(
                3,
                total_cols,
                hspace=0.30,
                wspace=0.10,
                left=0.05,
                right=0.95,
                top=0.85,
                bottom=0.1,
            )

            # Column 0: Original (reference)
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.imshow(orig_image_np)
            ax1.set_title("Image originale", fontsize=9, fontweight="bold")
            ax1.axis("off")
            for spine in ax1.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(2)
                spine.set_color("#333333")

            ax2 = fig.add_subplot(gs[1, 0])
            ax2.imshow(orig_mask_np, cmap="gray", vmin=0, vmax=1)
            ax2.set_title("Masque original", fontsize=10)
            ax2.axis("off")
            for spine in ax2.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1)
                spine.set_color("#333333")

            ax3 = fig.add_subplot(gs[2, 0])
            orig_overlay = orig_image_np.copy()
            orig_mask_colored = np.zeros_like(orig_overlay)
            orig_mask_colored[:, :, 1] = orig_mask_np * 255
            orig_composite = cv2.addWeighted(
                orig_overlay, 0.7, orig_mask_colored, 0.3, 0
            )
            ax3.imshow(orig_composite)

            orig_pixels = np.sum(orig_mask_np > 0)
            ax3.set_title(
                f"Image+Masque originaux\n{orig_pixels:,} pixels",
                fontsize=10,
            )
            ax3.axis("off")
            for spine in ax3.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1)  # Reduced from 2 to 1
                spine.set_color("#333333")

            # Columns 1+: Augmentations
            for aug_idx, (aug_name, aug_transform) in enumerate(augmentations):
                col = aug_idx + 1

                # Apply augmentation
                try:
                    augmented = aug_transform(image=orig_image_np, mask=orig_mask_np)
                    aug_image = augmented["image"]
                    aug_mask = augmented["mask"]
                except Exception as e:
                    logger.warning(f"Failed to apply {aug_name}: {e}")
                    aug_image = orig_image_np
                    aug_mask = orig_mask_np

                # Augmented image
                ax1 = fig.add_subplot(gs[0, col])
                ax1.imshow(aug_image)
                ax1.set_title(
                    f"{aug_name}", fontsize=9, fontweight="bold"
                )
                ax1.axis("off")
                for spine in ax1.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(2)  # Reduced from 3 to 2
                    spine.set_color("#333333")  # Use same color for all

                # Augmented mask
                ax2 = fig.add_subplot(gs[1, col])
                ax2.imshow(aug_mask, cmap="gray", vmin=0, vmax=1)
                ax2.set_title("Masque augmenté", fontsize=10)
                ax2.axis("off")
                for spine in ax2.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(1)
                    spine.set_color("#333333")

                # Augmented overlay
                ax3 = fig.add_subplot(gs[2, col])
                aug_overlay = aug_image.copy()
                aug_mask_colored = np.zeros_like(aug_overlay)
                aug_mask_colored[:, :, 1] = aug_mask * 255
                aug_composite = cv2.addWeighted(
                    aug_overlay, 0.7, aug_mask_colored, 0.3, 0
                )
                ax3.imshow(aug_composite)

                aug_pixels = np.sum(aug_mask > 0)

                ax3.set_title(
                    f"Augmentation\n{aug_pixels:,} pixels",
                    fontsize=10,
                )
                ax3.axis("off")
                for spine in ax3.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(1)
                    spine.set_color("#333333")

            # Save
            filename = f"augmentation_{category_name.lower().replace(' ', '_')}_sample_{sample_idx + 1}.png"
            save_path = os.path.join(graphics_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
            plt.close()

            logger.info(f"Saved: {filename}")


# ====================================
# Data Loading Helper
# ====================================
def load_fold_data(fold_files, images_dir, masks_dir):
    """Load data paths from fold files."""
    data_paths = []

    for fold_file in fold_files:
        with open(fold_file, "r") as f:
            filenames = [line.strip() for line in f.readlines()]

        images = []
        masks = []

        for filename in filenames:
            # Construct paths
            mask_path = os.path.join(masks_dir, filename)
            image_path = os.path.join(images_dir, filename)

            # Check if files exist
            if os.path.exists(image_path) and os.path.exists(mask_path):
                images.append(image_path)
                masks.append(mask_path)

        data_paths.append({"images": images, "masks": masks})
        logger.info(f"Loaded {len(images)} samples from {fold_file}")

    return data_paths


def load_test_data(test_file, images_dir, masks_dir):
    """Load test data paths."""
    with open(test_file, "r") as f:
        filenames = [line.strip() for line in f.readlines()]

    images = []
    masks = []

    for filename in filenames:
        mask_path = os.path.join(masks_dir, filename)
        image_path = os.path.join(images_dir, filename)

        if os.path.exists(image_path) and os.path.exists(mask_path):
            images.append(image_path)
            masks.append(mask_path)

    logger.info(f"Loaded {len(images)} test samples")
    return {"images": images, "masks": masks}

# ====================================
# Resume Function
# ====================================

def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    """Load checkpoint and return epoch, history, and other state."""
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Model state loaded")
    
    # Load optimizer state
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    logger.info("Optimizer state loaded")
    
    # Load scheduler state  
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    logger.info("Scheduler state loaded")
    
    # Extract other info
    start_epoch = checkpoint["epoch"] + 1  # Start from next epoch
    fold = checkpoint["fold"]
    best_val_iou = checkpoint["val_iou"]
    
    # Try to load history if it exists
    history = checkpoint.get("history", {
        "train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": [],
        "train_precision": [], "val_precision": [], "train_recall": [], "val_recall": [],
        "train_iou": [], "val_iou": [], "train_f1_score": [], "val_f1_score": [],
        "learning_rates": []
    })

    logger.info(f"Resuming from epoch {start_epoch}, fold {fold}")
    logger.info(f"Best validation IoU so far: {best_val_iou:.4f}")
    
    return start_epoch, fold, best_val_iou, history

# ====================================
# Main Training Function
# ====================================
def train_single_fold(
    fold,
    train_paths,
    val_paths,
    test_paths,
    config,
    save_dir,
    graphics_dir,
    device,
    debug_datasets,
    show_augmentation_examples,
    graphics_every_epoch,
    resume_checkpoint=None,
):
    """Train a single fold and evaluate on test set."""
    fold_start_time = time.time()

    cache_id, cleanup_cache = setup_fold_specific_cache(fold)
    
    logger.info(f"{'=' * 50}")
    logger.info(f"Training Fold {fold} - Started (Cache: {cache_id})")
    if resume_checkpoint:
        logger.info(f"Resuming from: {resume_checkpoint}")
    logger.info(f"{'=' * 50}")

    # auto_batch_size
    if config.get('auto_batch_size', False):
        logger.info("Auto batch size search enabled")
        
        # Create model and datasets for batch size testing
        test_model = create_model(config)
        test_model = optimize_model_memory_format(test_model, device)
        test_model = test_model.to(device)

        train_dataset_for_search = SimpleSegmentationDataset(
            train_paths["images"],
            train_paths["masks"],
            config["img_size"],
            get_transforms(True),
            cache_size=20,
        )
        
        val_dataset_for_search = SimpleSegmentationDataset(
            val_paths["images"],
            val_paths["masks"],
            config["img_size"],
            get_transforms(False),
            cache_size=20,
        )
        
        # Clean up test resources
        del test_model, train_dataset_for_search, val_dataset_for_search
        clear_memory()

    # Create datasets
    train_dataset = SimpleSegmentationDataset(
        train_paths["images"],
        train_paths["masks"],
        config["img_size"],
        get_transforms(True),
        cache_size=20,
    )
    val_dataset = SimpleSegmentationDataset(
        val_paths["images"],
        val_paths["masks"],
        config["img_size"],
        get_transforms(False),
        cache_size=20,
    )
    test_dataset = SimpleSegmentationDataset(
        test_paths["images"],
        test_paths["masks"],
        config["img_size"],
        get_transforms(False),
        cache_size=20,
    )

    # Debugging dataset samples
    if debug_datasets and fold == 0:
        logger.info("DEBUGGING TRAINING DATASET:")
        debug_dataset_samples(train_dataset, graphics_dir, 3)
        logger.info("DEBUGGING VALIDATION DATASET:")
        debug_dataset_samples(val_dataset, graphics_dir, 3)
        logger.info("DEBUGGING TEST DATASET:")
        debug_dataset_samples(test_dataset, graphics_dir, 2)

        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        logger.info(f"Test samples: {len(test_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )

    # Create model
    model = create_model(config)
    model = optimize_model_memory_format(model, device)
    model = model.to(device)
    param_count = count_parameters(model)

    # Mixed precision scaler
    scaler = GradScaler()

    # Loss function
    criterion = nn.BCEWithLogitsLoss()
    # optimizer = optim.Adam(
    #     model.parameters(),
    #     lr=config["learning_rate"],
    # )
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
        fused=True
    )

    # ReduceLROnPlateau scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.75,
        patience=10,
        min_lr=config["learning_rate"]/ 1000,
    )

    logger.info(f"Model: {config['architecture']} with {config['backbone']} backbone")
    logger.info(f"Model parameters: {param_count:,} ({param_count / 1e6:.2f}M)")
    logger.info(f"Encoder weights: {config['encoder_weights']}")
    logger.info(f"Device: {device}")
    logger.info("Mixed Precision: Enabled")
    logger.info(f"Image size: {config['img_size']}")
    logger.info(f"Base Learning rate: {config['learning_rate']}")

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "train_precision": [],
        "val_precision": [],
        "train_recall": [],
        "val_recall": [],
        "train_iou": [],
        "val_iou": [],
        "train_f1_score": [],
        "val_f1_score": [],
        "learning_rates": [],
    }

    start_epoch = 0
    best_val_iou = 0.0
    patience_counter = 0

    # Load checkpoint if resuming
    if resume_checkpoint:
        try:
            start_epoch, checkpoint_fold, best_val_iou, loaded_history = load_checkpoint(
                resume_checkpoint, model, optimizer, scheduler, device
            )
            
            # Use loaded history instead of fresh one
            history = loaded_history

            # Verify fold matches
            if checkpoint_fold != fold:
                logger.warning(f"Checkpoint fold ({checkpoint_fold}) != requested fold ({fold})")
                logger.warning("Continuing anyway, but verify this is intentional")
            
            logger.info(f"Successfully resumed from epoch {start_epoch}")
            logger.info(f"Loaded history with {len(history.get('train_loss', []))} previous epochs")

            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            logger.error("Starting fresh training instead")
            start_epoch = 0
            best_val_iou = 0.0

    # Training loop
    for epoch in range(start_epoch, config["epochs"]):
        epoch_start_time = time.time()
        logger.info(f"Epoch {epoch + 1}/{config['epochs']}")

        # Data augmentation visualizations
        if epoch == 0 and show_augmentation_examples:
            try:
                logger.info("PROGRESS creating data augmentation visualizations...")
                create_augmentation_report(train_dataset, graphics_dir, num_samples=5)

                logger.info("COMPLETED Data augmentation visualizations created successfully")
            except Exception as e:
                logger.warning(f"Could not create augmentation visualizations: {e}")

        # Train
        train_loss, train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            accumulation_steps=config.get("accumulation_steps", 1),
        )

        # Validate
        val_loss, val_metrics = validate_one_epoch(model, val_loader, criterion, device)

        # Apply scheduler - ReduceLROnPlateau
        try:
            old_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_metrics["iou"])
            current_lr = optimizer.param_groups[0]["lr"]

            # Log LR changes
            if current_lr != old_lr:
                logger.info(
                    f"Learning rate reduced from {old_lr:.6f} to {current_lr:.6f}"
                )
        except Exception as e:
            logger.warning(f"Error in scheduler step: {e}")
            current_lr = optimizer.param_groups[0]["lr"]

        # Store history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["learning_rates"].append(current_lr)
        for key in ["iou", "f1_score", "accuracy", "recall", "precision"]:
            if key in train_metrics and key in val_metrics:
                history[f"train_{key}"].append(train_metrics[key])
                history[f"val_{key}"].append(val_metrics[key])
            else:
                logger.warning(f"Missing metric: {key}")
                history[f"train_{key}"].append(0.0)
                history[f"val_{key}"].append(0.0)

        # Print progress
        logger.info(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        logger.info(
            f"  Train IoU: {train_metrics['iou']:.4f}, Val IoU: {val_metrics['iou']:.4f}"
        )
        logger.info(f"  Learning Rate: {current_lr:.6f}")

        # Graphics
        if epoch % 50 == 0:
            if graphics_every_epoch:
                save_epoch_graphics(
                    history, epoch, fold, graphics_dir, val_loader, model, device
                )

        # Save best model
        if val_metrics["iou"] > best_val_iou:
            best_val_iou = val_metrics["iou"]
            model_path = os.path.join(save_dir, f"best_model_fold_{fold}.pth")
            torch.save(model.state_dict(), model_path)
            logger.info(f"  New best model saved (IoU: {best_val_iou:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1

        # Save model every 50 epochs
        if epoch % 50 == 0:
            checkpoint_path = os.path.join(
                save_dir, f"checkpoint_epoch_{epoch}_fold_{fold}.pth"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "fold": fold,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_iou": val_metrics["iou"],
                    "history": history,
                },
                checkpoint_path,
            )

        # Calculate epoch time
        epoch_elapsed = time.time() - epoch_start_time
        epoch_time_str = f"{int(epoch_elapsed // 60):02d}:{int(epoch_elapsed % 60):02d}"

        # Enhanced progress logging
        fold_elapsed = time.time() - fold_start_time
        avg_epoch_time = fold_elapsed / (epoch + 1)
        remaining_epochs = config["epochs"] - (epoch + 1)
        eta_seconds = remaining_epochs * avg_epoch_time
        eta_str = f"{int(eta_seconds // 3600):02d}:{int((eta_seconds % 3600) // 60):02d}:{int(eta_seconds % 60):02d}"

        # Progress info
        progress_pct = ((epoch + 1) / config["epochs"]) * 100
        logger.info(
            f"Epoch {epoch + 1}/{config['epochs']} completed in {epoch_time_str}"
        )
        logger.info(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        logger.info(
            f"  Train IoU: {train_metrics['iou']:.4f}, Val IoU: {val_metrics['iou']:.4f}"
        )
        logger.info(
            f"  Progress: {progress_pct:.1f}% | ETA: {eta_str} | Best IoU: {best_val_iou:.4f}"
        )
        logger.info(
            f"Model: {config['architecture']} with {config['backbone']} backbone | Model parameters: {param_count / 1e6:.0f}M | Fold val: {fold}/5 |  Progress: {progress_pct:.1f}% | Best IoU: {best_val_iou:.4f} | Learning rate {current_lr:.6f} | Patience: {patience_counter}/{config['patience']}"
        )

        # Early stopping
        if patience_counter >= config["patience"]:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

        # Memory cleanup
        clear_memory()

    # Load best model for evaluation. handle case where best model might not exist
    best_model_path = os.path.join(save_dir, f"best_model_fold_{fold}.pth")
    if os.path.exists(best_model_path):
        logger.info(f"Loading best model from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path))
    else:
        logger.warning(f"Best model not found at {best_model_path}")
        logger.warning("Using current model state for evaluation")
        # Save current model state as fallback
        torch.save(model.state_dict(), best_model_path)
        logger.info(f"Saved current model state to {best_model_path}")

    # Final validation
    final_val_loss, final_val_metrics = validate_one_epoch(
        model, val_loader, criterion, device
    )

    # Evaluate on test set
    test_metrics = evaluate_on_test_set_smp(model, test_loader, device, save_dir)

    logger.info(
        f"FINAL Validation | Loss: {final_val_loss:.4f} | IoU: {final_val_metrics['iou']:.4f} | F1: {final_val_metrics['f1_score']:.4f} | Accuracy: {final_val_metrics['accuracy']:.4f} | Precision: {final_val_metrics['precision']:.4f} | Recall: {final_val_metrics['recall']:.4f}"
    )
    logger.info(
        f"FINAL Test Set | IoU: {test_metrics['iou']:.4f} | F1: {test_metrics['f1_score']:.4f} | Accuracy: {test_metrics['accuracy']:.4f} | Precision: {test_metrics['precision']:.4f} | Recall: {test_metrics['recall']:.4f}"
    )

    # Create visualizations
    plot_path = plot_training_history(history, fold, save_dir)
    pred_path = visualize_predictions(model, val_loader, device, fold, save_dir)

    logger.info(f"Fold {fold} completed!")
    fold_total_time = time.time() - fold_start_time
    fold_time_str = f"{int(fold_total_time // 3600):02d}:{int((fold_total_time % 3600) // 60):02d}:{int(fold_total_time % 60):02d}"
    logger.info(f"Fold {fold} completed in {fold_time_str}")
    logger.info(f"Best Validation IoU: {best_val_iou:.4f}")
    logger.info(f"Best Validation IoU: {best_val_iou:.4f}")
    logger.info("Final Validation Metrics:")
    for metric, value in final_val_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    logger.info("Test Set Metrics:")
    for metric, value in test_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")

    # Memory cleanup
    clear_memory()

    # Calculate comprehensive timing information
    fold_total_time = time.time() - fold_start_time
    fold_time_str = f"{int(fold_total_time // 3600):02d}:{int((fold_total_time % 3600) // 60):02d}:{int(fold_total_time % 60):02d}"

    # Add timing to return data
    return {
        "fold": fold,
        "best_val_iou": best_val_iou,
        "final_val_metrics": final_val_metrics,
        "test_metrics": test_metrics,
        "plot_path": plot_path,
        "pred_path": pred_path,
        "timing": {
            "total_seconds": fold_total_time,
            "total_time_formatted": fold_time_str,
            "total_epochs_trained": epoch + 1,
            "average_seconds_per_epoch": fold_total_time / (epoch + 1) if epoch >= 0 else 0,
            "start_timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(fold_start_time)),
            "end_timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
        }
    }

# ====================================
# One Fold Training Function
# ====================================
def train_single_fold_wrapper(
    config_name, fold, device, data_paths, test_paths, 
    debug_datasets, verbose_logging, show_augmentation_examples,
    graphics_every_epoch, resume_checkpoint=None, output_base_dir=None
):
    """Wrapper to train a single specific fold."""
    
    logger.info(f"Training single fold: {fold}")
    config = CONFIGS[config_name].copy()
    config["device"] = device
    
    # Generate timestamp for this single fold training
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if output_base_dir is None:
        output_base_dir = f"training_{datetime.datetime.now().strftime('%Y%m%d')}"
    base_model_dir = f"{output_base_dir}/{config['architecture']}_{config['backbone']}_{config['encoder_weights']}/single_fold_{fold}_{timestamp}"
    
    # Setup directories
    model_dir = os.path.join(base_model_dir, "01_model_output")
    log_dir = os.path.join(base_model_dir, "02_logs") 
    graphics_dir = os.path.join(base_model_dir, "03_graphics")
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(graphics_dir, exist_ok=True)
    
    # Setup logger
    setup_logger(log_dir, verbose=verbose_logging)
    
    # Prepare train/validation split
    val_paths = data_paths[fold]
    train_paths = {"images": [], "masks": []}
    for i, fold_data in enumerate(data_paths):
        if i != fold:
            train_paths["images"].extend(fold_data["images"])
            train_paths["masks"].extend(fold_data["masks"])
    
    logger.info(f"Training samples: {len(train_paths['images'])}")
    logger.info(f"Validation samples: {len(val_paths['images'])}")
    logger.info(f"Test samples: {len(test_paths['images'])}")
    
    # Train this fold
    fold_result = train_single_fold(
        fold=fold,
        train_paths=train_paths,
        val_paths=val_paths, 
        test_paths=test_paths,
        config=config,
        save_dir=model_dir,
        graphics_dir=graphics_dir,
        device=device,
        debug_datasets=debug_datasets,
        show_augmentation_examples=show_augmentation_examples,
        graphics_every_epoch=graphics_every_epoch,
        resume_checkpoint=resume_checkpoint
    )

    # Save comprehensive results:
    single_fold_results_file = os.path.join(model_dir, "single_fold_complete_results.json")
    with open(single_fold_results_file, "w") as f:
        json.dump(
            {
                "config": config_name,
                "fold": fold,
                "training_config": {
                    k: str(v) if not isinstance(v, (int, float, str, bool)) else v
                    for k, v in config.items()
                    if k != "device"
                },
                "results": fold_result,
                "timing_summary": fold_result.get("timing", {}),
                "performance_summary": {
                    "best_validation_iou": fold_result["best_val_iou"],
                    "test_iou": fold_result["test_metrics"]["iou"],
                    "training_duration": fold_result.get("timing", {}).get("total_time_formatted", "00:00:00"),
                    "epochs_completed": fold_result.get("timing", {}).get("total_epochs_trained", 0),
                }
            },
            f,
            indent=4,
        )  

    return fold_result

# ====================================
# Sequential All-Fold Training Function
# ====================================
def train_all_folds_sequential(
    config_name,
    device,
    data_paths,
    test_paths,
    debug_datasets,
    verbose_logging,
    show_augmentation_examples=False,
    graphics_every_epoch=True,
    output_base_dir=None,
):
    """Train using each fold as validation sequentially (0->1->2->3->4)."""
    all_folds_start_time = time.time()

    logger.info("=" * 60)
    logger.info("STARTING SEQUENTIAL ALL-FOLD TRAINING")
    logger.info("=" * 60)

    config = CONFIGS[config_name].copy()
    config["device"] = device

    # Generate timestamp ONCE for all folds
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if output_base_dir is None:
        output_base_dir = f"training_{datetime.datetime.now().strftime('%Y%m%d')}"
    base_model_dir = f"{output_base_dir}/{config['architecture']}_{config['backbone']}_{config['encoder_weights']}/all_folds_{timestamp}"

    all_results = []

    for validation_fold in range(5):
        fold_start_time = time.time()

        # Progress with ETA
        if validation_fold > 0:
            avg_fold_time = (time.time() - all_folds_start_time) / validation_fold
            remaining_folds = 5 - validation_fold
            eta_seconds = remaining_folds * avg_fold_time
            eta_str = f"{int(eta_seconds // 3600):02d}:{int((eta_seconds % 3600) // 60):02d}:{int(eta_seconds % 60):02d}"

            logger.info(
                f"Starting fold {validation_fold}/5 | ETA for all folds: {eta_str}"
            )

        logger.info(f"\n{'#' * 60}")
        logger.info(f"STARTING TRAINING WITH FOLD {validation_fold} AS VALIDATION")
        logger.info(f"{'#' * 60}")

        # Setup output directories for this fold (using the same timestamp)
        model_dir = os.path.join(
            f"{base_model_dir}/01_model_output", f"fold_{validation_fold}_{timestamp}"
        )
        log_dir = os.path.join(
            f"{base_model_dir}/02_logs", f"fold_{validation_fold}_{timestamp}"
        )
        graphics_dir = os.path.join(
            f"{base_model_dir}/03_graphics", f"fold_{validation_fold}_{timestamp}"
        )

        # Create directories
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(graphics_dir, exist_ok=True)

        # Setup logger for this fold
        setup_logger(log_dir, verbose=verbose_logging)

        # Prepare train/validation split
        val_paths = data_paths[validation_fold]

        # Use all other folds for training
        train_paths = {"images": [], "masks": []}
        for i, fold_data in enumerate(data_paths):
            if i != validation_fold:
                train_paths["images"].extend(fold_data["images"])
                train_paths["masks"].extend(fold_data["masks"])

        logger.info(f"Training samples: {len(train_paths['images'])}")
        logger.info(f"Validation samples: {len(val_paths['images'])}")
        logger.info(f"Test samples: {len(test_paths['images'])}")

        # Train this fold
        fold_result = train_single_fold(
            fold=validation_fold,
            train_paths=train_paths,
            val_paths=val_paths,
            test_paths=test_paths,
            config=config,
            save_dir=model_dir,
            graphics_dir=graphics_dir,
            device=device,
            debug_datasets=debug_datasets,
            show_augmentation_examples=show_augmentation_examples,
            graphics_every_epoch=graphics_every_epoch,
        )

        # Save fold results
        fold_summary_file = os.path.join(model_dir, "fold_training_summary.json")
        with open(fold_summary_file, "w") as f:
            json.dump(
                {
                    "config": config_name,
                    "validation_fold": validation_fold,
                    "training_config": {
                        k: str(v) if not isinstance(v, (int, float, str, bool)) else v
                        for k, v in config.items()
                        if k != "device"
                    },
                    "results": fold_result,
                    "fold_timing": fold_result.get("timing", {}),
                    "summary": {
                        "fold_duration_seconds": fold_result.get("timing", {}).get("total_seconds", 0),
                        "fold_duration_formatted": fold_result.get("timing", {}).get("total_time_formatted", "00:00:00"),
                        "epochs_completed": fold_result.get("timing", {}).get("total_epochs_trained", 0),
                        "avg_time_per_epoch_seconds": fold_result.get("timing", {}).get("average_seconds_per_epoch", 0),
                    }
                },
                f,
                indent=4,
            )

        # Log fold completion with timing
        fold_elapsed = time.time() - fold_start_time
        fold_time_str = f"{int(fold_elapsed // 3600):02d}:{int((fold_elapsed % 3600) // 60):02d}:{int(fold_elapsed % 60):02d}"

        logger.info(f"COMPLETED FOLD {validation_fold} in {fold_time_str}")
        logger.info(f"Best Validation IoU: {fold_result['best_val_iou']:.4f}")
        logger.info(f"Test IoU: {fold_result['test_metrics']['iou']:.4f}")

        all_results.append(fold_result)

        logger.info(f"COMPLETED FOLD {validation_fold}")
        logger.info(f"Best Validation IoU: {fold_result['best_val_iou']:.4f}")
        logger.info(f"Test IoU: {fold_result['test_metrics']['iou']:.4f}")

        # Memory cleanup between folds
        clear_memory()

        # Log progress
        logger.info(f"PROGRESS: Completed {validation_fold + 1}/5 folds")

    # Calculate overall statistics
    val_ious = [result["best_val_iou"] for result in all_results]
    test_ious = [result["test_metrics"]["iou"] for result in all_results]

    # Extract timing information
    fold_times = [result.get("timing", {}).get("total_seconds", 0) for result in all_results]
    total_training_time = sum(fold_times)
    avg_fold_time = total_training_time / len(fold_times) if fold_times else 0

    # Calculate overall training time
    total_time = time.time() - all_folds_start_time
    total_time_str = f"{int(total_time // 3600):02d}:{int((total_time % 3600) // 60):02d}:{int(total_time % 60):02d}"

    overall_stats = {
        "validation_ious": val_ious,
        "test_ious": test_ious,
        "mean_validation_iou": np.mean(val_ious),
        "std_validation_iou": np.std(val_ious),
        "mean_test_iou": np.mean(test_ious),
        "std_test_iou": np.std(test_ious),
        "all_fold_results": all_results,
        # Add comprehensive timing statistics
        "timing_summary": {
            "total_training_seconds": total_time,
            "total_training_time_formatted": total_time_str,
            "individual_fold_times_seconds": fold_times,
            "average_fold_time_seconds": avg_fold_time,
            "average_fold_time_formatted": f"{int(avg_fold_time // 3600):02d}:{int((avg_fold_time % 3600) // 60):02d}:{int(avg_fold_time % 60):02d}",
            "fastest_fold_seconds": min(fold_times) if fold_times else 0,
            "slowest_fold_seconds": max(fold_times) if fold_times else 0,
            "start_timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(all_folds_start_time)),
            "end_timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
            "total_folds_completed": len(all_results)
        }
    }

    # Save overall results
    overall_results_file = os.path.join(base_model_dir, "overall_results.json")
    with open(overall_results_file, "w") as f:
        json.dump(overall_stats, f, indent=4)

    # Final summary with total time
    total_time = time.time() - all_folds_start_time
    total_time_str = f"{int(total_time // 3600):02d}:{int((total_time % 3600) // 60):02d}:{int(total_time % 60):02d}"

    logger.info("\n" + "=" * 60)
    logger.info(f"ALL FOLDS TRAINING COMPLETED in {total_time_str}")
    logger.info("=" * 60)

    logger.info("FINAL RESULTS SUMMARY:")
    logger.info(f"Validation IoU: {np.mean(val_ious):.4f} ± {np.std(val_ious):.4f}")
    logger.info(f"Test IoU: {np.mean(test_ious):.4f} ± {np.std(test_ious):.4f}")
    logger.info("\nIndividual Fold Results:")
    for i, result in enumerate(all_results):
        logger.info(
            f"  Fold {i}: Val IoU={result['best_val_iou']:.4f}, Test IoU={result['test_metrics']['iou']:.4f}"
        )

    return overall_stats


def train_multiple_configs_sequential(
    config_names,
    device,
    data_paths,
    test_paths,
    debug_datasets,
    verbose_logging,
    show_augmentation_examples,
    graphics_every_epoch,
    output_base_dir=None,
):
    """Train multiple configurations sequentially - minimal version."""
    logger.info("=" * 80)
    logger.info("STARTING MULTI-CONFIGURATION SEQUENTIAL TRAINING")
    logger.info("=" * 80)
    logger.info(f"Configurations to train: {config_names}")
    logger.info(f"Total configurations: {len(config_names)}")

    all_config_results = {}

    for config_idx, config_name in enumerate(config_names):
        logger.info(f"\n{'#' * 80}")
        logger.info(
            f"STARTING CONFIGURATION {config_idx + 1}/{len(config_names)}: {config_name}"
        )
        logger.info("Training parameters:")
        # Create temporary model to count parameters
        temp_config = CONFIGS[config_name].copy()
        temp_model = create_model(temp_config)
        param_count = count_parameters(temp_model)
        del temp_model  # Clean up

        logger.info(f"  Model size: {param_count / 1e6:.2f}M parameters")
        logger.info(f"  Architecture: {CONFIGS[config_name]['architecture']}")
        logger.info(f"  Backbone: {CONFIGS[config_name]['backbone']}")
        logger.info(f"  Encoder Weights: {CONFIGS[config_name]['encoder_weights']}")
        logger.info(f"  Img size: {CONFIGS[config_name]['img_size']}")
        logger.info(f"  Batch size: {CONFIGS[config_name]['batch_size']}")
        logger.info(f"  Learning rate: {CONFIGS[config_name]['learning_rate']}")
        logger.info(f"  Epochs: {CONFIGS[config_name]['epochs']}")
        logger.info(f"  Patience: {CONFIGS[config_name]['patience']}")
        logger.info(f"{'#' * 80}")

        try:
            # Train all folds for this configuration
            config_results = train_all_folds_sequential(
                config_name,
                device,
                data_paths,
                test_paths,
                debug_datasets,
                verbose_logging,
                show_augmentation_examples,
                graphics_every_epoch,
            )
            all_config_results[config_name] = config_results

            logger.info(f"COMPLETED CONFIGURATION: {config_name}")
            logger.info(
                f"  Final Mean Validation IoU: {config_results['mean_validation_iou']:.4f} ± {config_results['std_validation_iou']:.4f}"
            )
            logger.info(
                f"  Final Mean Test IoU: {config_results['mean_test_iou']:.4f} ± {config_results['std_test_iou']:.4f}"
            )

        except Exception as e:
            logger.error(f"ERROR in configuration {config_name}: {e}")
            all_config_results[config_name] = {
                "error": str(e),
                "mean_validation_iou": 0.0,
                "mean_test_iou": 0.0,
            }

        # Memory cleanup between configurations
        clear_memory()

        # Progress update
        logger.info(
            f"MULTI-CONFIG PROGRESS: Completed {config_idx + 1}/{len(config_names)} configurations"
        )

    # Save basic results summary
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if output_base_dir is None:
        output_base_dir = f"training_{datetime.datetime.now().strftime('%Y%m%d')}"
    results_dir = f"{output_base_dir}/multi_config_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    results_file = os.path.join(results_dir, "all_config_results.json")
    with open(results_file, "w") as f:
        json.dump(all_config_results, f, indent=4)

    # Log final comparison
    logger.info("\n" + "=" * 80)
    logger.info("MULTI-CONFIGURATION TRAINING COMPLETED")
    logger.info("=" * 80)
    logger.info("FINAL SUMMARY:")
    logger.info("-" * 80)

    # Sort configurations by validation IoU
    valid_configs = {k: v for k, v in all_config_results.items() if "error" not in v}
    sorted_configs = sorted(
        valid_configs.items(), key=lambda x: x[1]["mean_validation_iou"], reverse=True
    )

    logger.info("Results (ranked by Validation IoU):")
    for rank, (config_name, results) in enumerate(sorted_configs, 1):
        logger.info(f"  #{rank}: {config_name}")
        logger.info(
            f"      Val IoU: {results['mean_validation_iou']:.4f} ± {results['std_validation_iou']:.4f}"
        )
        logger.info(
            f"      Test IoU: {results['mean_test_iou']:.4f} ± {results['std_test_iou']:.4f}"
        )

    # Log any failed configurations
    failed_configs = [k for k, v in all_config_results.items() if "error" in v]
    if failed_configs:
        logger.warning(f"\nFailed configurations: {failed_configs}")

    logger.info(f"\nResults saved to: {results_file}")

    return all_config_results


# ====================================
# Debug functions
# ====================================
def debug_dataset_samples(dataset, graphics_dir, num_samples=5):
    """Debug dataset to check if masks contain valid data - show original and processed."""
    logger.info("DEBUGGING DATASET SAMPLES")
    logger.info("=" * 50)

    for i in range(min(num_samples, len(dataset))):
        # Get processed data from dataset
        image, mask = dataset[i]

        # Load original images directly for comparison
        orig_image = Image.open(dataset.image_paths[i]).convert("RGB")
        orig_mask = Image.open(dataset.mask_paths[i])
        if orig_mask.mode != "L":
            orig_mask = orig_mask.convert("L")
        orig_mask = np.array(orig_mask)

        logger.info(f"Sample {i + 1}:")
        logger.info(f"  Original image size: {orig_image.size}")
        logger.info(f"  Original mask shape: {orig_mask.shape}")
        logger.info(f"  Original mask min/max: {orig_mask.min()}/{orig_mask.max()}")
        logger.info(f"  Original mask unique values: {np.unique(orig_mask)}")

        logger.info(f"  Processed image shape: {image.shape}")
        logger.info(f"  Processed image min/max: {image.min():.3f}/{image.max():.3f}")
        logger.info(f"  Processed mask shape: {mask.shape}")
        logger.info(f"  Processed mask min/max: {mask.min():.3f}/{mask.max():.3f}")
        logger.info(f"  Processed mask unique values: {torch.unique(mask)}")
        logger.info(f"  Processed mask positive pixels: {(mask > 0).sum().item()}")
        logger.info(f"  Processed mask total pixels: {mask.numel()}")
        logger.info(
            f"  Processed mask positive ratio: {(mask > 0).sum().item() / mask.numel():.4f}"
        )

        # Check if mask is completely empty
        if (mask > 0).sum().item() == 0:
            logger.warning(f"Processed mask {i + 1} is completely empty!")

        # Create 2x2 subplot: original and processed versions
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"Sample {i + 1}: Original vs Processed", fontsize=16)

        # Original image
        axes[0, 0].imshow(np.array(orig_image))
        axes[0, 0].set_title(f"Original Image {orig_image.size}")
        axes[0, 0].axis("off")

        # Original mask
        # Create binary version of original mask for better visualization
        orig_mask_binary = (
            (orig_mask > 127).astype(np.uint8) if orig_mask.max() > 1 else orig_mask
        )
        axes[0, 1].imshow(orig_mask_binary, cmap="gray", vmin=0, vmax=1)
        axes[0, 1].set_title(
            f"Original Mask (Pos: {np.sum(orig_mask_binary > 0)}, Max: {orig_mask.max()})"
        )
        axes[0, 1].axis("off")

        # Processed image
        img_np = image.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)
        axes[1, 0].imshow(img_np)
        axes[1, 0].set_title(f"Processed Image {dataset.img_size}")
        axes[1, 0].axis("off")

        # Processed mask
        mask_np = mask.numpy()
        im = axes[1, 1].imshow(mask_np, cmap="gray", vmin=0, vmax=1)
        axes[1, 1].set_title(
            f"Processed Mask (Pos: {(mask > 0).sum().item()}, Min: {mask_np.min():.3f}, Max: {mask_np.max():.3f})"
        )
        axes[1, 1].axis("off")

        # Colorbar to processed mask
        plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        save_path = os.path.join(graphics_dir, f"debug_sample_{i + 1}_comparison.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"  Saved comparison: {save_path}")

        # Check if resizing is causing issues
        orig_positive = np.sum(orig_mask_binary > 0)
        processed_positive = (mask > 0).sum().item()
        if orig_positive > 0 and processed_positive == 0:
            logger.error(
                f"  CRITICAL: Original mask has {orig_positive} positive pixels but processed has 0!"
            )
            logger.error("  This suggests the resizing is destroying the mask data!")
        elif orig_positive > 0:
            ratio_change = processed_positive / orig_positive
            logger.info(
                f"  Positive pixel ratio change: {ratio_change:.3f} (original: {orig_positive}, processed: {processed_positive})"
            )


# ====================================
# Command Line Interface
# ====================================
def parse_arguments():
    """Parse command line arguments - supports multiple configurations."""
    parser = argparse.ArgumentParser(
        description="Segmentation Training with Mixed Precision"
    )

    parser.add_argument(
        "--config",
        type=str,
        nargs="+",  # Allow multiple configurations
        required=True,
        choices=list(CONFIGS.keys()),
        help="Training configuration(s). Use multiple configs like: --config config1 config2 config3",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use (default: 0)")

    # Single fold training
    parser.add_argument("--fold", type=int, choices=[0, 1, 2, 3, 4], 
                       help="Train only specific fold (0-4). If not specified, trains all folds sequentially")

    # Batch size override
    parser.add_argument(
        "--batch_size", 
        type=int, 
        help="Override batch size from config (default: use config value)"
    )    

    # Resume training
    parser.add_argument("--resume", type=str, 
                       help="Path to checkpoint file to resume training from")

    # Output directory of training results
    parser.add_argument("--output_dir", type=str, 
                   default=f"training_{datetime.datetime.now().strftime('%Y%m%d')}", 
                   help="Base output directory for training results")

    parser.add_argument(
        "--debug_dataset",
        action="store_true",
        help="Debug dataset samples (default: False)",
    )
    parser.add_argument(
        "--verbose_logging",
        action="store_true",
        help="Enable debug logging (default: False)",
    )
    parser.add_argument(
        "--show_augmentation_examples",
        action="store_true",
        help="Create augmentation visualizations (default: False)",
    )
    parser.add_argument(
        "--no_graphics_every_epoch",
        action="store_true",
        help="Disable graphics every epoch (default: saves graphics every epoch)",
    )

    return parser.parse_args()


# ====================================
# Main Function
# ====================================
def main():
    # show number of cpu cores available and memory available
    logger.info(f"Available CPU cores: {os.cpu_count()}")
    logger.info(f"Available memory: {psutil.virtual_memory().total / (1024 ** 3):.2f} GB")

    # Log script start
    logger.info("=" * 80)
    logger.info(
        f"SCRIPT STARTED at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(SCRIPT_START_TIME))}"
    )
    logger.info("=" * 80)

    # Parse command line arguments
    args = parse_arguments()

    if args.batch_size is not None:
        if args.batch_size <= 0:
            logger.error(f"Invalid batch_size: {args.batch_size}. Must be > 0")
            return
        
        logger.info(f"Overriding batch_size to {args.batch_size} for all configurations")
        for config_name in args.config:
            original_batch_size = CONFIGS[config_name]["batch_size"]
            CONFIGS[config_name]["batch_size"] = args.batch_size
            logger.info(f"  {config_name}: {original_batch_size} -> {args.batch_size}")

    # Validate configurations exist
    for config_name in args.config:
        if config_name not in CONFIGS:
            logger.error(f"Configuration '{config_name}' not found in CONFIGS")
            logger.error(f"Available configurations: {list(CONFIGS.keys())}")
            return

    # Setup device
    if torch.cuda.is_available() and args.gpu < torch.cuda.device_count():
        device = torch.device(f"cuda:{args.gpu}")
        logger.info(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")

        # Enable optimizations for mixed precision
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        # Check if GPU supports mixed precision
        if torch.cuda.get_device_capability(args.gpu)[0] >= 7:
            logger.info("GPU supports Tensor Cores")
        else:
            logger.info(
                "GPU doesn't support Tensor Cores - Mixed precision may have limited benefits"
            )
    else:
        device = torch.device("cpu")
        logger.warning("Using CPU (CUDA not available or invalid GPU ID)")

    # Log configuration
    logger.info("Starting Segmentation Training")
    logger.info(f"Configuration(s): {args.config}")
    
    # Update training mode logging based on fold argument
    if args.fold is not None:
        logger.info(f"Training mode: SINGLE FOLD TRAINING - Fold {args.fold}")
        if args.resume:
            logger.info(f"Resume from: {args.resume}")
    else:
        logger.info("Training mode: AUTOMATIC ALL-FOLD SEQUENTIAL")
    
    logger.info(f"GPU: {args.gpu}")
    logger.info(f"Device: {device}")
    logger.info(f"Data directory: {BASE_DATASET_DIR}")

    # Load fold files
    fold_files = []
    for i in range(5):
        fold_file = os.path.join(BASE_DATASET_DIR, f"fold_{i}_dataset.txt")
        if os.path.exists(fold_file):
            fold_files.append(fold_file)
        else:
            logger.warning(f"Fold file not found: {fold_file}")

    if len(fold_files) != 5:
        logger.error(f"Expected 5 fold files, found {len(fold_files)}")
        return

    logger.info(f"Found {len(fold_files)} fold files")

    # Load data
    logger.info("Loading fold data...")
    data_paths = load_fold_data(fold_files, DATASET_IMAGES_DIR, DATASET_MASKS_DIR)

    logger.info("Loading test data...")
    test_paths = load_test_data(
        TEST_DATASET_FILE, DATASET_IMAGES_DIR, DATASET_MASKS_DIR
    )

    # Train based on mode and number of configurations
    try:
        # Check if single fold training is requested
        if args.fold is not None:
            logger.info(f"SINGLE FOLD TRAINING MODE - Fold {args.fold}")
            
            if len(args.config) > 1:
                logger.error("Single fold training supports only one configuration at a time")
                logger.error("Please specify only one --config when using --fold")
                return
            
            config_name = args.config[0]
            
            # Log configuration details
            logger.info(f"Configuration: {config_name}")
            logger.info("Training parameters:")
            # Create temporary model to count parameters
            temp_config = CONFIGS[config_name].copy()
            temp_model = create_model(temp_config)
            param_count = count_parameters(temp_model)
            del temp_model  # Clean up
            logger.info(f"  Model size: {param_count / 1e6:.2f}M parameters")
            config = CONFIGS[config_name].copy()
            for key, value in config.items():
                logger.info(f"  {key}: {value}")
            
            # Train single fold
            fold_result = train_single_fold_wrapper(
                config_name=config_name,
                fold=args.fold,
                device=device,
                data_paths=data_paths,
                test_paths=test_paths,
                debug_datasets=args.debug_dataset,
                verbose_logging=args.verbose_logging,
                show_augmentation_examples=args.show_augmentation_examples,
                graphics_every_epoch=not args.no_graphics_every_epoch,
                resume_checkpoint=args.resume,
                output_base_dir=args.output_dir
            )
            
            logger.info("=" * 60)
            logger.info(f"FOLD {args.fold} TRAINING COMPLETED!")
            logger.info("=" * 60)
            logger.info(f"Best Validation IoU: {fold_result['best_val_iou']:.4f}")
            logger.info(f"Test IoU: {fold_result['test_metrics']['iou']:.4f}")
            logger.info("Final Validation Metrics:")
            for metric, value in fold_result['final_val_metrics'].items():
                logger.info(f"  {metric}: {value:.4f}")
            logger.info("Test Set Metrics:")
            for metric, value in fold_result['test_metrics'].items():
                logger.info(f"  {metric}: {value:.4f}")
            
        elif len(args.config) == 1:
            # Single configuration training - ALL FOLDS
            config_name = args.config[0]
            config = CONFIGS[config_name].copy()
            config["device"] = device

            logger.info(f"Configuration: {config_name}")
            logger.info("Training parameters:")
            # Create temporary model to count parameters
            temp_config = CONFIGS[config_name].copy()
            temp_model = create_model(temp_config)
            param_count = count_parameters(temp_model)
            del temp_model  # Clean up
            logger.info(f"  Model size: {param_count / 1e6:.2f}M parameters")
            for key, value in config.items():
                if key != "device":
                    logger.info(f"  {key}: {value}")

            logger.info("Starting single configuration training...")
            overall_results = train_all_folds_sequential(
                config_name,
                device,
                data_paths,
                test_paths,
                args.debug_dataset,
                args.verbose_logging,
                args.show_augmentation_examples,
                not args.no_graphics_every_epoch,
                output_base_dir=args.output_dir
            )

            logger.info("=" * 60)
            logger.info("TRAINING COMPLETED!")
            logger.info("=" * 60)
            logger.info(
                f"Final Validation IoU: {overall_results['mean_validation_iou']:.4f} ± {overall_results['std_validation_iou']:.4f}"
            )
            logger.info(
                f"Final Test IoU: {overall_results['mean_test_iou']:.4f} ± {overall_results['std_test_iou']:.4f}"
            )

        else:
            # Multiple configuration training - ALL FOLDS
            logger.info(
                f"Starting multi-configuration training with {len(args.config)} configurations..."
            )
            all_config_results = train_multiple_configs_sequential(
                args.config,
                device,
                data_paths,
                test_paths,
                args.debug_dataset,
                args.verbose_logging,
                args.show_augmentation_examples,
                not args.no_graphics_every_epoch,
                output_base_dir=args.output_dir
            )

            logger.info("=" * 80)
            logger.info("ALL TRAINING COMPLETED!")
            logger.info("=" * 80)

            # Show best configuration
            valid_configs = {
                k: v for k, v in all_config_results.items() if "error" not in v
            }
            if valid_configs:
                best_config = max(
                    valid_configs.items(), key=lambda x: x[1]["mean_validation_iou"]
                )
                logger.info(f"BEST CONFIGURATION: {best_config[0]}")
                logger.info(
                    f"  Validation IoU: {best_config[1]['mean_validation_iou']:.4f} ± {best_config[1]['std_validation_iou']:.4f}"
                )
                logger.info(
                    f"  Test IoU: {best_config[1]['mean_test_iou']:.4f} ± {best_config[1]['std_test_iou']:.4f}"
                )
            else:
                logger.error("No configurations completed successfully!")

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        return

    # Log script completion
    total_script_time = get_elapsed_time()
    logger.info("=" * 80)
    logger.info(f"SCRIPT COMPLETED - Total time: {total_script_time}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
