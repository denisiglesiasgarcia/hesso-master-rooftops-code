# Geneva Rooftop Segmentation Pipeline (Identification des parties utiles des toitures pour le solaire par machine learning)

A comprehensive pipeline for processing aerial imagery and cadastral data to create high-quality datasets for rooftop segmentation in Geneva, Switzerland.

## Overview

This project provides an end-to-end workflow for preparing and processing geospatial data from SITG (Système d'Information du Territoire à Genève) to enable machine learning-based rooftop segmentation. The pipeline handles everything from raw cadastral data processing to the creation of analysis-ready image tiles with corresponding masks.

## Downloads

[Datasets and report](https://hessoit-my.sharepoint.com/:f:/g/personal/denis_iglesias_hes-so_ch/EmgLTRi4VoREhKbOVP2LmY8BA5MAFRORwOBu5rhdE6WaGg?e=oHkDxU)

## Data structure

```bash
# Notebooks
02_traitement_gpkg_classification.ipynb
03_download_orthophoto_geotiff.ipynb
04_geotiff_tiling_with_buildings.ipynb
05_dataset_selection.ipynb
06_annotations_postprocessing.ipynb
07_split_dataset.ipynb
08_prepare_dataset_yolo.ipynb
09_generate_slurm_configs.ipynb
10_evaluation_models.ipynb

# Data
data/
├── SITG/
│   ├── CAD_BATIMENT_HORSOL_TOIT_2024-11-03.gpkg
│   ├── CAD_BATIMENT_HORSOL_TOIT_SP_2024-11-03.gpkg
│   ├── CAD_BATIMENT_HORSOL_2024-11-03.gpkg
│   ├── CAD_COMMUNE_2024-11-03.gpkg
│   └── ortho2019/
└── notebook_XX/

# Datasets
datasets/
└── supervisely/
    ├── dataset_processed_20250523-173715 # Dataset SMP goes here
    └── yolo_processed_20250619_151249    # Dataset YOLO goes here

# This readme
README.md

# requirements for the different environments needed
requirements/
├── rooftops_myenv.yml
├── rooftops_yolo.yml
└── rooftops_pytorch.yml

# training scripts
training_configs.py
training_smp.py
training_yolo.py
```

## Notebooks

### GPKG File Processing and Classification (`02_traitement_gpkg_classification.ipynb`)

Download, verify, transform and visualize GPKG (Geopackage) files for automatic roof classification.

**Workflow:**

1. Load and validate GPKG data with geometry verification
2. Apply spatial filtering and data quality checks
3. Implement automatic roof classification based on defined criteria (not used in the methodology)
4. Generate visualizations and export results (not used in the methodology)

### Orthophoto Download and Processing (`03_download_orthophoto_geotiff.ipynb`)

Download SITG orthophotos, generate tiles and visualizations.

**Workflow:**

1. URL discovery and validation for orthophoto archives
2. Parallel download management
3. Extraction and processing
4. Tile generation and coverage analysis

### GeoTIFF Tiling and Building Detection (`04_geotiff_tiling_with_buildings.ipynb`)

Process SITG orthophotos to generate tiles containing building footprints.

**Workflow:**

1. Load GeoTIFF orthophotos and building footprint data
2. Split large GeoTIFFs into manageable tiles with buffers
3. Detect buildings within each tile using spatial indexing
4. Generate metadata for tiles containing buildings

### Dataset Selection (`05_dataset_selection.ipynb`)

Select and prepare a balanced dataset from classified rooftops for analysis.

**Workflow:**

1. Load classified rooftop data and GeoTIFF metadata
2. Enrich data by joining rooftop classifications with tile information
3. Create balanced dataset sampling across SIA categories and area bins
4. Generate visualizations and export dataset for further processing

### Postprocessing Annotations Supervisely (`06_annotations_postprocessing.ipynb`)

Process annotated rooftop segmentation data for quality control and standardization.

**Workflow:**

1. Load dataset and annotation metadata
2. Clip images using building polygons
3. Handle tile overlaps and remove duplicates
4. Standardize dimensions to 1280x1280
5. Validate data integrity and export

### Annotated Dataset Split into Train/Val/Test (`07_split_dataset.ipynb`)

Split annotated tile dataset into stratified k-fold cross-validation and test sets.

**Workflow:**

1. Load processed annotated dataset with dominant classes and area bins
2. Apply iterative stratification to ensure balanced distribution
3. Create k-fold cross-validation splits plus final test set
4. Validate distribution quality and data leakage prevention
5. Generate comprehensive visualizations and export dataset files

### YOLO Cross validation datasets (`08_prepare_dataset_yolo.ipynb`)

Convert Supervisely CV dataset to YOLO format with data augmentation

**Workflow:**

1. Convert binary masks to YOLO segmentation format
2. Create cross-validation datasets with augmentation

### Generate slurm sbatch files for the SMP models (`09_generate_slurm_configs.ipynb`)

Generate sbatch files for slurm that can automate sending jobs, script to watch progress of trainings and script to resume failed trainings.

**Workflow:**

1. Generate sbatch files with resume per encoder-decoder
2. Script to send jobs per size/per fold
3. Script to follow all the trainings
4. Script to resume failed trainings

### Model Evaluation (`10_evaluation_models.ipynb`)

Evaluate segmentation models on test dataset and analyze performance.

**Workflow:**

1. Load trained models and configurations
2. Run inference on test dataset
3. Calculate comprehensive metrics (IoU, mAP, F1-score)
4. Generate visualizations and performance analysis
5. Create ensemble predictions from k-fold models

### Report Visualizations (`99_report_visualizations.ipynb`)

Visualizations for the report in latex

## Python scripts for training

### Models configurations (`training_configs.py`)

All the configurations for the models.

```python
IMG_SIZE = (1280, 1280)
NUM_CLASSES = 1
EPOCHS = 1000
PATIENCE = 50
LEARNING_RATE = 0.001
ACCUMULATION_STEPS = 4

BATCH_SIZE_SMALL = 4
BATCH_SIZE_MEDIUM = 2
BATCH_SIZE_LARGE = 1
BATCH_SIZE_HUGE = 2

BATCH_SIZE_MIN = 1
BATCH_SIZE_MAX = 128
BATCH_SIZE_NB_TESTS = 2

CONFIGS = {
    "unet_efficientnet_b3_imagenet": {                # model name encoder_decoder_dataset
        "architecture": "unet",                       # decoder
        "backbone": "timm-efficientnet-b3",           # encoder
        "encoder_weights": "imagenet",                # pretrained weights (always imagenet)
        "img_size": IMG_SIZE,                         # image size (always 1280x1280)
        "num_classes": NUM_CLASSES,                   # The number of classes is always 1
        "learning_rate": LEARNING_RATE,               # Learning rate, most cases 0.001
        "epochs": EPOCHS,                             # Default 1000 epochs
        "patience": PATIENCE,                         # Stop the training if the model does not improve after n epoch, default 50
        "accumulation_steps": ACCUMULATION_STEPS,     # Gradient accumulation for simulating bigger batch size
        "auto_batch_size": False,                     # Automatic batch size search, doesn't work in slurm clusters. Default deactivated
        "batch_size": BATCH_SIZE_SMALL,               # Batch size
        "min_batch_size_search": BATCH_SIZE_MIN,      # Minimum batch size for the batch size search. Default deactivated
        "max_batch_size_search": BATCH_SIZE_MAX,      # Maximum batch size for the batch size search. Default deactivated
        "batch_size_test_steps": BATCH_SIZE_NB_TESTS, # Number of training with validation to do in the batch size search. Default deactivated
    },
    ...
}
```

### Train SMP model (`training_smp.py`)

```bash
python3 training_smp.py
    --config unet_mambaout_small_imagenet         # model name from CONFIGS in training_configs.py
    --fold 0                                      # fold number 0,1,2,3,4. Default trains all folds
    --gpu 0                                       # GPU number, default 0
    --output_dir 01_training_medium_unet_20250626 # output folder for the training
    --batch_size 4                                # batch size from CONFIGS in training_configs.py
    --no_graphics_every_epoch                     # removes graphics during training for faster training
    --verbose_logging                             # Verbose mode for the logger
```

### Train YOLO model (`training_yolo.py`)

```bash
python3 training_yolo.py
    --model yolo12n-seg.yaml                                           # model name: yolo12n-seg.yaml, yolo12s-seg.yaml, yolo12m-seg.yaml, yolo12l-seg.yaml, yolo12x-seg.yaml
    --folds 4                                                          # fold number 0,1,2,3,4. Default trains all folds
    --batch-size 8                                                     # Batch size
    --output-dir 01_training_yolo12n_20250623                          # Output folder for the training 
    --dataset-path datasets/supervisely/yolo_processed_20250619_151249 # Yolo dataset path

```

## Requirements

```bash
# main env
conda env create -f requirements/rooftops_myenv.yml

# yolo
conda env create -f requirements/rooftops_yolo.yml

# pytorch
conda env create -f requirements/rooftops_pytorch.yml
```