# train_yolo.py

import shutil
import time
from datetime import datetime
from pathlib import Path
import argparse

import numpy as np
import yaml
from loguru import logger
import torch
from ultralytics import YOLO
import cv2
import json


def calculate_metrics_standalone(pred_logits, target, threshold=0.5):
    """Calculate segmentation metrics efficiently using PyTorch operations.
    
    Args:
        pred_logits: Raw model outputs (logits/probabilities) with shape [B, H, W] or [B, 1, H, W]
        target: Binary ground truth masks with shape [B, H, W] or [B, 1, H, W]
        threshold: Threshold for converting probabilities to binary predictions
    
    Returns:
        dict: Dictionary containing precision, recall, accuracy, f1_score, and iou metrics
    """
    try:
        # Ensure tensors are on the same device
        device = pred_logits.device
        target = target.to(device)
        
        # Normalize tensor dimensions by removing channel dimension if present
        if len(pred_logits.shape) == 4 and pred_logits.shape[1] == 1:
            pred_logits = pred_logits.squeeze(1)
        if len(target.shape) == 4 and target.shape[1] == 1:
            target = target.squeeze(1)
            
        # Convert logits to probabilities using sigmoid if values are outside [0,1]
        if pred_logits.max() > 1.0 or pred_logits.min() < 0.0:
            pred_probs = torch.sigmoid(pred_logits)
        else:
            pred_probs = pred_logits
        
        # Convert to binary predictions
        pred_binary = (pred_probs > threshold).float()
        target_binary = (target > 0.5).float()
        
        # Validate input data
        if torch.isnan(pred_binary).any() or torch.isnan(target_binary).any():
            logger.warning("NaN values detected in predictions or targets")
            return {
                'precision': 0.0, 'recall': 0.0, 'accuracy': 0.0, 
                'f1_score': 0.0, 'iou': 0.0
            }
        
        # Flatten tensors for metric calculation
        pred_flat = pred_binary.flatten()
        target_flat = target_binary.flatten()
        
        # Calculate confusion matrix components
        tp = (pred_flat * target_flat).sum()
        fp = (pred_flat * (1 - target_flat)).sum()
        fn = ((1 - pred_flat) * target_flat).sum()
        tn = ((1 - pred_flat) * (1 - target_flat)).sum()
        
        # Calculate metrics with epsilon to prevent division by zero
        epsilon = 1e-8
        
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        accuracy = (tp + tn) / (tp + fp + fn + tn + epsilon)
        f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
        iou = tp / (tp + fp + fn + epsilon)
        
        # Convert to Python floats
        metrics = {
            'precision': float(precision.item()),
            'recall': float(recall.item()),
            'accuracy': float(accuracy.item()),
            'f1_score': float(f1_score.item()),
            'iou': float(iou.item())
        }
        
        # Replace any inf/nan values with 0.0
        for key, value in metrics.items():
            if not np.isfinite(value):
                metrics[key] = 0.0
                
        return metrics
        
    except Exception as e:
        logger.error(f"Error in metrics calculation: {e}")
        return {
            'precision': 0.0, 'recall': 0.0, 'accuracy': 0.0, 
            'f1_score': 0.0, 'iou': 0.0
        }


class YOLOSegmentationTrainer:
    """Simplified YOLO segmentation training pipeline for single or multiple folds.
    
    This class provides a streamlined training pipeline that relies on YOLO's built-in
    metrics and plotting capabilities without additional comprehensive analysis.
    """
    
    def __init__(self, 
                 dataset_processed_path: str,
                 model_name: str = "yolo11n-seg.pt",
                 project_name: str = "rooftop_segmentation",
                 output_dir: str = None,
                 device: str = "auto"):
        """Initialize the YOLO segmentation trainer.
        
        Args:
            dataset_processed_path: Path to processed CV datasets
            model_name: Pre-trained segmentation model to use
            project_name: Project name for saving results
            output_dir: Custom directory to save training results (defaults to current directory)
            device: Device to use for training ('auto', 'cpu', '0', '1', etc.)
        """
        self.dataset_processed_path = Path(dataset_processed_path)
        self.model_name = model_name
        self.project_name = project_name
        self.device = device
        
        # Configure output directory structure
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.full_project_path = self.output_dir / project_name
        else:
            self.output_dir = Path(".")
            self.full_project_path = Path(project_name)
        
        # Default training parameters
        self.training_params = {
            'epochs': 100,
            'imgsz': 640,
            'batch': 16,
        }
        
        # Initialize result storage
        self.cv_results = []
        
        logger.info("Initialized YOLO Segmentation Trainer (Simplified)")
        logger.info(f"Model: {model_name}")
        logger.info(f"Device: {device}")
        logger.info(f"Dataset path: {dataset_processed_path}")
        logger.info(f"Output directory: {self.full_project_path}")
        
        # Validate dataset path
        if not self.dataset_processed_path.exists():
            logger.error(f"Dataset path not found: {self.dataset_processed_path}")
        else:
            logger.info(f"Dataset path validated: {self.dataset_processed_path}")

    def _cleanup_project_directory(self):
        """Remove existing project directory to prevent auto-increment naming issues."""
        if self.full_project_path.exists():
            logger.info(f"Cleaning up existing project directory: {self.full_project_path}")
            shutil.rmtree(self.full_project_path)
            time.sleep(1)
            logger.info("Project directory cleaned up")

    def _cleanup_gpu_memory(self):
        """Clear GPU memory cache between training folds."""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("GPU memory cache cleared")
        except Exception as e:
            logger.error(f"Failed to clear GPU memory: {e}")

    def create_dataset_yaml(self, fold_path: Path, class_names: list = None):
        """Create YOLO dataset configuration file for the specified fold.
        
        Args:
            fold_path: Path to specific fold dataset
            class_names: List of class names (defaults to ["free_space"])
        
        Returns:
            Path: Path to created YAML configuration file
        """
        if class_names is None:
            class_names = ["free_space"]
        
        dataset_config = {
            'path': str(fold_path.absolute()),
            'train': 'train/images',
            'val': 'val/images', 
            'test': 'test/images',
            'nc': len(class_names),
            'names': {i: name for i, name in enumerate(class_names)}
        }
        
        yaml_path = fold_path / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        logger.info(f"Created dataset config: {yaml_path}")
        return yaml_path

    def evaluate_detailed_metrics(self, model, dataset_path, fold_num, stage="validation"):
        """Calculate detailed IoU and other metrics on validation or test set.
        
        Args:
            model: Trained YOLO model
            dataset_path: Path to dataset
            fold_num: Fold number for identification
            stage: Evaluation stage ("validation" or "test")
        
        Returns:
            dict: Dictionary containing detailed evaluation metrics including IoU
        """
        logger.info(f"Calculating detailed {stage} metrics for fold {fold_num}")
        
        # Determine image directory based on stage
        if stage == "validation":
            images_dir = dataset_path / "val" / "images"
        else:  # test
            images_dir = dataset_path / "test" / "images"
        
        # Search for mask directory
        mask_dir_candidates = [
            dataset_path / stage / "masks",
            dataset_path / "val" / "masks" if stage == "validation" else dataset_path / "test" / "masks",
            dataset_path / "masks",
            dataset_path.parent / "masks"
        ]
        
        masks_dir = None
        for candidate in mask_dir_candidates:
            if candidate.exists():
                masks_dir = candidate
                break
        
        if not images_dir.exists() or masks_dir is None:
            logger.warning(f"Dataset directories not found: {images_dir} or masks directory")
            return {
                'precision': 0.0, 'recall': 0.0, 'accuracy': 0.0, 
                'f1_score': 0.0, 'iou': 0.0, 'sample_count': 0
            }
        
        # Build valid image-mask pairs
        image_paths = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
        
        valid_pairs = []
        for img_path in image_paths:
            mask_name = img_path.stem + ".png"
            mask_path = masks_dir / mask_name
            if mask_path.exists():
                valid_pairs.append((str(img_path), str(mask_path)))
            else:
                # Try alternative mask file extensions
                for ext in [".jpg", ".jpeg", ".bmp"]:
                    mask_path = masks_dir / (img_path.stem + ext)
                    if mask_path.exists():
                        valid_pairs.append((str(img_path), str(mask_path)))
                        break
        
        if len(valid_pairs) == 0:
            logger.warning(f"No valid image-mask pairs found for {stage} evaluation")
            return {
                'precision': 0.0, 'recall': 0.0, 'accuracy': 0.0, 
                'f1_score': 0.0, 'iou': 0.0, 'sample_count': 0
            }
        
        logger.info(f"Found {len(valid_pairs)} valid pairs for {stage} evaluation")
        
        # Perform batch prediction and evaluation
        image_files = [pair[0] for pair in valid_pairs]
        mask_files = [pair[1] for pair in valid_pairs]
        
        all_metrics = []
        
        try:
            # Use YOLO's batch prediction
            results = model.predict(
                source=image_files,
                verbose=False,
                save=False,
                stream=True,
                imgsz=self.training_params['imgsz']
            )
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            for idx, (result, mask_file) in enumerate(zip(results, mask_files)):
                try:
                    # Load and preprocess ground truth mask
                    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
                    if mask is None:
                        continue
                        
                    # Resize mask to match YOLO output dimensions
                    mask = cv2.resize(mask, (self.training_params['imgsz'], self.training_params['imgsz']), 
                                    interpolation=cv2.INTER_NEAREST)
                    
                    # Normalize mask to binary values
                    if mask.max() > 1:
                        mask = mask.astype(np.float32) / 255.0
                        mask = (mask > 0.5).astype(np.float32)
                    else:
                        mask = (mask > 0).astype(np.float32)
                    
                    mask_tensor = torch.from_numpy(mask).to(device)
                    
                    # Extract prediction mask from YOLO result
                    if hasattr(result, 'masks') and result.masks is not None:
                        pred_mask = result.masks.data[0]
                        
                        # Ensure prediction dimensions match target
                        if pred_mask.shape != mask_tensor.shape:
                            pred_mask = torch.nn.functional.interpolate(
                                pred_mask.unsqueeze(0).unsqueeze(0), 
                                size=mask_tensor.shape, 
                                mode='bilinear', 
                                align_corners=False
                            ).squeeze()
                        
                        pred_mask = pred_mask.to(device)
                    else:
                        # Handle case where no prediction is made
                        pred_mask = torch.zeros_like(mask_tensor)
                    
                    # Calculate detailed metrics
                    batch_metrics = calculate_metrics_standalone(
                        pred_mask.unsqueeze(0), mask_tensor.unsqueeze(0)
                    )
                    all_metrics.append(batch_metrics)
                    
                except Exception as e:
                    logger.error(f"Error processing sample {idx}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error in batch prediction: {e}")
            return {
                'precision': 0.0, 'recall': 0.0, 'accuracy': 0.0, 
                'f1_score': 0.0, 'iou': 0.0, 'sample_count': 0
            }
        
        if not all_metrics:
            logger.warning(f"No metrics calculated for {stage}")
            return {
                'precision': 0.0, 'recall': 0.0, 'accuracy': 0.0, 
                'f1_score': 0.0, 'iou': 0.0, 'sample_count': 0
            }
        
        # Calculate average metrics across all samples
        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            avg_metrics[key] = float(np.mean(values))
        
        avg_metrics['sample_count'] = len(all_metrics)
        
        # Log evaluation results
        logger.info(f"Detailed {stage} metrics for fold {fold_num}:")
        for metric, value in avg_metrics.items():
            if metric != 'sample_count':
                logger.info(f"  {metric}: {value:.4f}")
        logger.info(f"  Samples evaluated: {avg_metrics['sample_count']}")
        
        return avg_metrics

    def train_single_fold(self, 
                        fold_num: int, 
                        class_names: list = None,
                        custom_params: dict = None):
        """Train YOLO model on a single fold.
        
        Args:
            fold_num: Fold number (0-4)
            class_names: List of class names for the dataset
            custom_params: Custom training parameters to override defaults
        
        Returns:
            dict: Training results and metadata for the fold
        """
        logger.info(f"Starting training for fold {fold_num}")
        
        # Validate fold dataset path
        fold_path = self.dataset_processed_path / f"fold_{fold_num}_dataset"
        
        if not fold_path.exists():
            raise ValueError(f"Fold dataset not found: {fold_path}")
        
        # Use existing dataset configuration or create new one
        yaml_path = fold_path / 'dataset.yaml'
        
        # Recreate the yaml file to ensure correct paths
        if yaml_path.exists():
            logger.info(f"Removing existing dataset config: {yaml_path}")
            yaml_path.unlink()

        yaml_path = self.create_dataset_yaml(fold_path, class_names)
        logger.info(f"Created new dataset config with correct paths: {yaml_path}")
        
        # Initialize pre-trained model
        model = YOLO(self.model_name)
        logger.info(f"Loaded pre-trained model: {self.model_name}")
        
        # Configure training parameters
        train_params = self.training_params.copy()
        if custom_params:
            train_params.update(custom_params)
        
        expected_output_dir = self.full_project_path / f"fold_{fold_num}"
        
        # Clean up any existing fold directory
        if expected_output_dir.exists():
            logger.info(f"Removing existing fold directory: {expected_output_dir}")
            shutil.rmtree(expected_output_dir)
            time.sleep(0.5)
        
        # Ensure parent directory exists
        self.full_project_path.mkdir(parents=True, exist_ok=True)
        
        # Execute model training
        start_time = time.time()
        
        try:
            logger.info("Starting YOLO training with built-in metrics and plots")
            results = model.train(
                data=str(yaml_path),
                project=str(self.full_project_path),
                name=f"fold_{fold_num}",
                device=self.device,
                exist_ok=False,
                verbose=True,
                **train_params
            )
            
            training_time = time.time() - start_time
            
            # Perform final validation to extract metrics
            logger.info("Running YOLO validation to extract final metrics")
            val_results = model.val(data=str(yaml_path), verbose=True)
            
            # Determine actual save directory
            if hasattr(results, 'save_dir'):
                actual_save_dir = str(results.save_dir)
                logger.info(f"YOLO saved results to: {actual_save_dir}")
            else:
                actual_save_dir = str(expected_output_dir)
                logger.warning(f"Could not get save_dir from results, using expected: {actual_save_dir}")
            
            # Extract YOLO metrics
            try:
                if hasattr(val_results, 'seg'):
                    # Extract segmentation-specific metrics
                    yolo_metrics = {
                        'mAP50': float(val_results.seg.map50),
                        'mAP50-95': float(val_results.seg.map),
                        'precision': float(val_results.seg.mp),
                        'recall': float(val_results.seg.mr),
                    }
                    
                    # Attempt to extract IoU if available
                    if hasattr(val_results.seg, 'iou'):
                        yolo_metrics['iou'] = float(val_results.seg.iou)
                    elif hasattr(val_results, 'maps'):
                        if len(val_results.maps) > 0:
                            yolo_metrics['map_per_class'] = float(np.mean(val_results.maps))
                
                elif hasattr(val_results, 'box'):
                    # Fallback to detection metrics if segmentation unavailable
                    logger.warning(f"Using detection metrics for fold {fold_num} - ensure segmentation model is used")
                    yolo_metrics = {
                        'mAP50': float(val_results.box.map50),
                        'mAP50-95': float(val_results.box.map),
                        'precision': float(val_results.box.mp),
                        'recall': float(val_results.box.mr),
                    }
                else:
                    # Default fallback metrics
                    logger.warning(f"Could not extract YOLO metrics for fold {fold_num}")
                    yolo_metrics = {
                        'mAP50': 0.0, 'mAP50-95': 0.0, 'precision': 0.0, 'recall': 0.0,
                    }
                
                # Extract training information
                total_epochs = len(getattr(results, 'results', [])) if hasattr(results, 'results') else getattr(results, 'epochs', 0)
                best_epoch = getattr(results, 'best_epoch', total_epochs)
                best_fitness = float(getattr(results, 'best_fitness', 0.0))
                
                # Calculate detailed IoU metrics on validation and test sets
                logger.info("Calculating detailed validation metrics...")
                detailed_val_metrics = self.evaluate_detailed_metrics(model, fold_path, fold_num, "validation")
                
                logger.info("Calculating detailed test metrics...")
                detailed_test_metrics = self.evaluate_detailed_metrics(model, fold_path, fold_num, "test")
                
                # Compile fold results with detailed metrics
                fold_results = {
                    'fold': fold_num,
                    'training_time': training_time,
                    'total_epochs': total_epochs,
                    'best_epoch': best_epoch,
                    'best_fitness': best_fitness,
                    'yolo_metrics': yolo_metrics,
                    'detailed_val_metrics': detailed_val_metrics,
                    'detailed_test_metrics': detailed_test_metrics,
                    'model_path': str(Path(actual_save_dir) / "weights" / "best.pt"),
                    'results_dir': actual_save_dir,
                    'fold_path': str(fold_path),
                }
                
                # Display training results
                logger.info("=" * 80)
                logger.info(f"FOLD {fold_num} COMPLETE RESULTS")
                logger.info("=" * 80)
                logger.info(f"Training completed in: {training_time/60:.1f} minutes")
                logger.info(f"Total epochs trained: {total_epochs}")
                logger.info(f"Best epoch: {best_epoch}")
                logger.info(f"Best fitness: {best_fitness:.4f}")
                
                logger.info("\nYOLO Built-in Metrics:")
                for metric_name, metric_value in yolo_metrics.items():
                    if metric_name == 'mAP50':
                        logger.info(f"  mAP@0.5: {metric_value:.4f} (Primary metric)")
                    elif metric_name == 'mAP50-95':
                        logger.info(f"  mAP@0.5:0.95: {metric_value:.4f}")
                    elif metric_name == 'precision':
                        logger.info(f"  Precision: {metric_value:.4f}")
                    elif metric_name == 'recall':
                        logger.info(f"  Recall: {metric_value:.4f}")
                    elif metric_name == 'iou':
                        logger.info(f"  IoU (YOLO): {metric_value:.4f}")
                    elif metric_name == 'map_per_class':
                        logger.info(f"  mAP per class: {metric_value:.4f}")
                    else:
                        logger.info(f"  {metric_name}: {metric_value:.4f}")
                
                logger.info("\nDetailed Validation Metrics:")
                for metric, value in detailed_val_metrics.items():
                    if metric != 'sample_count':
                        logger.info(f"  {metric}: {value:.4f}")
                logger.info(f"  Validation samples: {detailed_val_metrics.get('sample_count', 0)}")
                
                logger.info("\nDetailed Test Metrics:")
                for metric, value in detailed_test_metrics.items():
                    if metric != 'sample_count':
                        logger.info(f"  {metric}: {value:.4f}")
                logger.info(f"  Test samples: {detailed_test_metrics.get('sample_count', 0)}")
                
                # Highlight key IoU metrics
                val_iou = detailed_val_metrics.get('iou', 0.0)
                test_iou = detailed_test_metrics.get('iou', 0.0)
                logger.info(f"\nKey IoU Metrics:")
                logger.info(f"  Validation IoU: {val_iou:.4f}")
                logger.info(f"  Test IoU: {test_iou:.4f}")
                
                logger.info(f"\nModel saved: {fold_results['model_path']}")
                logger.info(f"Results saved to: {actual_save_dir}")
                logger.info("=" * 80)
                
            except Exception as e:
                logger.error(f"Error extracting results for fold {fold_num}: {e}")
                
                # Try to calculate detailed metrics even if YOLO metrics failed
                try:
                    logger.info("Attempting to calculate detailed metrics despite YOLO metrics error...")
                    detailed_val_metrics = self.evaluate_detailed_metrics(model, fold_path, fold_num, "validation")
                    detailed_test_metrics = self.evaluate_detailed_metrics(model, fold_path, fold_num, "test")
                except Exception as detailed_error:
                    logger.error(f"Failed to calculate detailed metrics: {detailed_error}")
                    detailed_val_metrics = {'precision': 0.0, 'recall': 0.0, 'accuracy': 0.0, 'f1_score': 0.0, 'iou': 0.0, 'sample_count': 0}
                    detailed_test_metrics = {'precision': 0.0, 'recall': 0.0, 'accuracy': 0.0, 'f1_score': 0.0, 'iou': 0.0, 'sample_count': 0}
                
                # Create minimal fallback results
                fold_results = {
                    'fold': fold_num,
                    'training_time': training_time,
                    'total_epochs': 0,
                    'best_epoch': 0,
                    'best_fitness': 0.0,
                    'yolo_metrics': {'mAP50': 0.0, 'mAP50-95': 0.0, 'precision': 0.0, 'recall': 0.0},
                    'detailed_val_metrics': detailed_val_metrics,
                    'detailed_test_metrics': detailed_test_metrics,
                    'model_path': str(Path(actual_save_dir) / "weights" / "best.pt"),
                    'results_dir': actual_save_dir,
                    'fold_path': str(fold_path),
                }
            
            # Save basic metrics
            self._save_basic_metrics(fold_results, actual_save_dir)
            
            self.cv_results.append(fold_results)
            
            logger.info(f"Fold {fold_num} training completed successfully")
            
            return fold_results
            
        except Exception as e:
            logger.error(f"Training failed for fold {fold_num}: {e}")
            raise
            
        finally:
            # Clean up resources
            try:
                del model
                self._cleanup_gpu_memory()
            except Exception as cleanup_error:
                logger.error(f"Error during cleanup after fold {fold_num}: {cleanup_error}")

    def _save_basic_metrics(self, fold_results, save_dir):
        """Save basic training metrics to JSON file."""
        try:
            metrics_file = Path(save_dir) / "training_metrics.json"
            
            # Prepare data for JSON serialization
            metrics_data = {
                'fold': fold_results['fold'],
                'training_time_minutes': fold_results['training_time'] / 60,
                'total_epochs': fold_results['total_epochs'],
                'best_epoch': fold_results['best_epoch'],
                'best_fitness': fold_results['best_fitness'],
                'yolo_metrics': fold_results['yolo_metrics'],
                'detailed_validation_metrics': fold_results['detailed_val_metrics'],
                'detailed_test_metrics': fold_results['detailed_test_metrics'],
                'model_path': fold_results['model_path'],
                'timestamp': datetime.now().isoformat()
            }
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=4)
            
            logger.info(f"Training metrics saved: {metrics_file}")
            
        except Exception as e:
            logger.error(f"Failed to save training metrics: {e}")

    def train_folds(self, 
                   folds: list = None,
                   class_names: list = None,
                   custom_params: dict = None):
        """Train YOLO model on specified folds.
        
        Args:
            folds: List of fold numbers to train (defaults to [0,1,2,3,4])
            class_names: List of class names for the dataset
            custom_params: Custom training parameters to override defaults
        """
        if folds is None:
            folds = list(range(5))
        
        logger.info(f"Starting training for folds: {folds}")
        
        # Clean up any existing project directory
        self._cleanup_project_directory()
        
        # Train each specified fold
        training_start_time = time.time()
        successful_folds = []
        
        for fold_num in folds:
            try:
                fold_result = self.train_single_fold(fold_num, class_names, custom_params)
                successful_folds.append(fold_num)
                logger.info(f"Successfully completed fold {fold_num}")
            except Exception as e:
                logger.error(f"Failed to train fold {fold_num}: {e}")
                continue
        
        training_end_time = time.time()
        total_duration = training_end_time - training_start_time
        
        # Log summary
        logger.info("=" * 80)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Requested folds: {folds}")
        logger.info(f"Successfully trained: {successful_folds}")
        logger.info(f"Failed folds: {[f for f in folds if f not in successful_folds]}")
        logger.info(f"Total training time: {total_duration/60:.1f} minutes")
        
        if self.cv_results:
            # Calculate average metrics across successful folds
            avg_map50 = np.mean([r['yolo_metrics']['mAP50'] for r in self.cv_results])
            avg_precision = np.mean([r['yolo_metrics']['precision'] for r in self.cv_results])
            avg_recall = np.mean([r['yolo_metrics']['recall'] for r in self.cv_results])
            
            # Calculate average IoU metrics
            avg_val_iou = np.mean([r['detailed_val_metrics']['iou'] for r in self.cv_results if r['detailed_val_metrics']])
            avg_test_iou = np.mean([r['detailed_test_metrics']['iou'] for r in self.cv_results if r['detailed_test_metrics']])
            avg_val_f1 = np.mean([r['detailed_val_metrics']['f1_score'] for r in self.cv_results if r['detailed_val_metrics']])
            avg_test_f1 = np.mean([r['detailed_test_metrics']['f1_score'] for r in self.cv_results if r['detailed_test_metrics']])
            
            logger.info(f"\nAverage Performance Across {len(self.cv_results)} Folds:")
            logger.info(f"  Average mAP@0.5: {avg_map50:.4f}")
            logger.info(f"  Average Precision: {avg_precision:.4f}")
            logger.info(f"  Average Recall: {avg_recall:.4f}")
            logger.info(f"  Average Validation IoU: {avg_val_iou:.4f}")
            logger.info(f"  Average Test IoU: {avg_test_iou:.4f}")
            logger.info(f"  Average Validation F1: {avg_val_f1:.4f}")
            logger.info(f"  Average Test F1: {avg_test_f1:.4f}")
            
            # Save training summary
            self._save_training_summary(folds, successful_folds, total_duration)
        
        logger.info("=" * 80)
        logger.info("Training completed")

    def _save_training_summary(self, requested_folds, successful_folds, total_duration):
        """Save overall training summary."""
        try:
            summary_file = self.full_project_path / "training_summary.json"
            
            summary_data = {
                'requested_folds': requested_folds,
                'successful_folds': successful_folds,
                'failed_folds': [f for f in requested_folds if f not in successful_folds],
                'total_training_time_minutes': total_duration / 60,
                'fold_results': self.cv_results,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(summary_file, 'w') as f:
                json.dump(summary_data, f, indent=4, default=str)
            
            logger.info(f"Training summary saved: {summary_file}")
            
        except Exception as e:
            logger.error(f"Failed to save training summary: {e}")


def main():
    """Main function with command line argument parsing and training execution."""
    
    # Define available model configurations
    model_list = [
        "yolo12n-seg.yaml",
        "yolo12s-seg.yaml", 
        "yolo12m-seg.yaml",
        "yolo12l-seg.yaml",
        "yolo12x-seg.yaml"
    ]
    
    # Configure command line argument parser
    parser = argparse.ArgumentParser(
        description="Simplified YOLO Segmentation Training for Single or Multiple Folds"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        choices=model_list,
        required=True,
        help="YOLO model configuration to train"
    )
    parser.add_argument(
        "--folds", 
        type=int, 
        nargs="+", 
        choices=range(5),
        default=list(range(5)),
        help="Fold numbers to train (0-4). Default: all folds"
    )
    parser.add_argument(
        "--dataset-path", 
        type=str, 
        default="datasets/supervisely/yolo_processed_20250619_151249",
        help="Path to processed dataset directory"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="training_yolo",
        help="Output directory for training results"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="0",
        help="Device to use for training (auto, cpu, 0, 1, etc.)"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=1000,
        help="Maximum number of training epochs"
    )
    parser.add_argument(
        "--image-size", 
        type=int, 
        default=1280,
        help="Training image size (square)"
    )
    parser.add_argument(
        "--patience", 
        type=int, 
        default=25,
        help="Early stopping patience (epochs)"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=None,
        help="Batch size for training. If not specified, uses model-specific defaults"
    )
    
    args = parser.parse_args()
    
    # Extract configuration parameters
    MODEL_NAME = args.model
    todaysdate = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Model-specific default batch size configuration
    DEFAULT_BATCH_SIZE = {
        "yolo12n-seg.yaml": 8,
        "yolo12s-seg.yaml": 4,
        "yolo12m-seg.yaml": 2,
        "yolo12l-seg.yaml": 2,
        "yolo12x-seg.yaml": 1,
    }
    
    # Determine batch size (user-specified or model default)
    BATCH_SIZE = args.batch_size if args.batch_size is not None else DEFAULT_BATCH_SIZE[MODEL_NAME]
    
    # Configure training parameters
    DATASET_PROCESSED_PATH = args.dataset_path
    OUTPUT_DIR_YOLO = args.output_dir
    PROJECT_NAME = f"yolo_{MODEL_NAME.split('.')[0]}_{todaysdate}"
    CLASS_NAMES = ["free_space"]
    DEVICE = args.device
    
    CUSTOM_PARAMS = {
    # Core training parameters
    'epochs': args.epochs,
    'batch': BATCH_SIZE,
    'imgsz': args.image_size,
    'patience': args.patience,
    
    # Learning rate optimization
     'lr0': 0.001,
    'lrf': 0.01,
    'warmup_epochs': 5,
    'warmup_bias_lr': 0.1,
    'warmup_momentum': 0.8,
    
    # Optimizer settings
    'optimizer': 'AdamW',    # Try AdamW instead of SGD
    
    # Loss function weights (critical for segmentation)
    'box': 7.5,              # Box loss weight
    'cls': 0.5,              # Classification loss weight
    'dfl': 1.5,              # DFL loss weight
    # Note: seg loss weight is handled internally by YOLO
    
    # Validation and saving
    'val': True,
    'plots': True,
    'save_period': 50,
    
    # Hardware optimization
    'amp': True,
    'half': True,
    'workers': 2,
    'cache': False,
    
    # Keep augmentation disabled as requested
    "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,
    "degrees": 0.0, "translate": 0.0, "scale": 0.0,
    "shear": 0.0, "perspective": 0.0, "flipud": 0.0,
    "fliplr": 0.0, "bgr": 0.0, "mosaic": 0.0,
    "mixup": 0.0, "cutmix": 0.0, "copy_paste": 0.0,
    "erasing": 0.0, "close_mosaic": False,
    "auto_augment": None,
    }

    # CUSTOM_PARAMS = {
    #     'epochs': args.epochs,
    #     'batch': BATCH_SIZE,
    #     'imgsz': args.image_size,
    #     'patience': args.patience,
    #     'lr0': 0.01,
    #     'warmup_epochs': 3,
    #     'plots': True,  # Enable YOLO's built-in plots
    #     'save_period': 50,
        
    #     # DataLoader optimization
    #     'workers': 2,
    #     'cache': False,

    #     # GPU acceleration for Ampere architecture
    #     'amp': True,
    #     'half': True,

    #     # Data augmentation disabled for consistency
    #     "hsv_h": 0.0,
    #     "hsv_s": 0.0,
    #     "hsv_v": 0.0,
    #     "degrees": 0.0,
    #     "translate": 0.0,
    #     "scale": 0.0,
    #     "shear": 0.0,
    #     "perspective": 0.0,
    #     "flipud": 0.0,
    #     "fliplr": 0.0,
    #     "bgr": 0.0,
    #     "mosaic": 0.0,
    #     "mixup": 0.0,
    #     "cutmix": 0.0,
    #     "copy_paste": 0.0,
    #     "erasing": 0.0,
    #     'close_mosaic': False,
    #     "auto_augment": None,
    # }
    
    # Configure GPU optimizations for Ampere architecture
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    
    # Log training configuration
    logger.info("Simplified YOLO Segmentation Training")
    logger.info("Configuration:")
    logger.info(f"  Model: {MODEL_NAME}")
    logger.info(f"  Folds: {args.folds}")
    logger.info(f"  Dataset: {DATASET_PROCESSED_PATH}")
    logger.info(f"  Output: {OUTPUT_DIR_YOLO}")
    logger.info(f"  Device: {DEVICE}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Image size: {args.image_size}")
    logger.info(f"  Batch size: {BATCH_SIZE} {'(custom)' if args.batch_size is not None else '(default)'}")
    logger.info(f"  Patience: {args.patience}")
    logger.info("  YOLO built-in plots and metrics enabled")
    
    # Initialize trainer
    trainer = YOLOSegmentationTrainer(
        dataset_processed_path=DATASET_PROCESSED_PATH,
        model_name=MODEL_NAME,
        project_name=PROJECT_NAME,
        output_dir=OUTPUT_DIR_YOLO,
        device=DEVICE
    )

    # Execute training
    logger.info(f"Starting training for folds: {args.folds}")
    trainer.train_folds(
        folds=args.folds,
        class_names=CLASS_NAMES,
        custom_params=CUSTOM_PARAMS
    )


if __name__ == "__main__":
    main()