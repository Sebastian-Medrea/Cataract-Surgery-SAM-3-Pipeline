# Cataract Surgery Tool Tracking: SAM 3 vs YOLOv26

CISC 499 Undergraduate Project

Sebastian Medrea, supervised by Dr. Rebecca Hisey

## About

This project compares two different deep learning methods for automatically tracking surgical instruments in simulated cataract surgery videos. The goal is to see if we can streamline surgical skill assessment with a cheaper, less invasive alternative to EM tracking, and whether these models can automate some of the annotation process to reduce the burden of expert labelling.

The two main methods are:

1. **SAM 3** (Segment Anything Model 3) from Meta. This is a zero-shot approach where I give the model one bounding box on the first frame a tool shows up, and it tracks the tool through the rest of the video on its own. No training needed.

2. **YOLOv26** trained with Leave-One-Out Cross Validation. Each fold holds out one participant for testing and trains on the other four. This is the standard supervised method that needs actual ground truth labels.

There is also a third experiment where I train YOLO using SAM 3's predictions as pseudo-labels instead of ground truth. The point is to see if SAM 3 can generate usable training data on its own.

All three are evaluated against ground truth bounding boxes using mAP, precision, recall, and F1. I also compute motion metrics (path length and usage time) from the predicted bounding boxes and run statistical tests to see if the tracking can distinguish expert surgeons from novices.

## Dataset

Data comes from the **PerkTutor CataractSurgery** simulated dataset on the Perk Lab network drive:

```
P:/data/PerkTutor/CataractSurgery/
    Videos/Simulated_Data/mkv/
    Datasets/Simulated_Data/
```

5 participants (P1 through P5), 5 trials each, so 25 videos total at 15 FPS. P1 is the expert ophthalmologist and P2 through P5 are novice residents. Ground truth CSVs have timestamped bounding boxes for 6 tool classes. P1 has revised labels so there is separate path handling for that in the code.

The dataset is not included in this repo. You need access to the Perk Lab drive or a local copy. Update `base_path` in `config.yaml` to point to your copy.

## Files

| File | Description |
|------|-------------|
| `run.py` | Main CLI entry point for running any pipeline |
| `sam3_pipeline.py` | SAM 3 zero-shot tracking pipeline |
| `yolo26_pipeline.py` | YOLOv26 training and inference with ground truth labels |
| `yolo26_sam3_combined.py` | YOLOv26 trained on SAM 3 pseudo-labels |
| `results_analysis.py` | Statistical analysis and all the plots |
| `utils.py` | Shared utilities (IoU, mAP, CSV export, motion metrics, config loading) |
| `config.yaml` | Central configuration file (paths, model settings, participants) |
| `environment.yml` | Conda environment specification |
| `login_helper.py` | Logs you into HuggingFace for SAM 3 model weights |

## Scripts

### run.py

Main entry point. Provides a CLI to run individual pipelines or the full workflow without editing any source files. All settings come from `config.yaml` by default and can be overridden with command-line flags.

```bash
python run.py sam3                          # Run SAM 3 zero-shot tracking (all participants)
python run.py sam3 --single                 # Quick test on one video only
python run.py sam3 --mode sequential        # Override tracking mode
python run.py yolo                          # Run YOLO GT-trained LOOCV
python run.py yolo --epochs 50              # Train YOLO with 50 epochs
python run.py yolo --participants P1 P2     # Only run on P1 and P2
python run.py yolo-sam3                     # Train YOLO on SAM 3 pseudo-labels
python run.py analyze                       # Run statistical analysis and plotting
python run.py analyze --output ./results    # Save analysis to custom directory
python run.py all                           # Run full pipeline end to end
```

Each pipeline module (`sam3_pipeline.py`, `yolo26_pipeline.py`, etc.) can also be run directly with `python sam3_pipeline.py` if preferred, but `run.py` is the recommended way.

### sam3_pipeline.py

Loads the SAM 3 model from HuggingFace in bfloat16 and runs zero-shot bounding box tracking on the cataract videos. It reads ground truth CSVs to find where each tool first appears, gives SAM 3 that initial bounding box as a prompt, and then SAM 3 propagates through the video in chunks (default 100 frames). The output masks get converted back to bounding boxes and compared against ground truth.

Two tracking modes:
- `track_sequential` tracks one tool at a time, resets the model memory between tools
- `track_simultaneous` tracks all tools in one pass with multi-object tracking

The mode and chunk size are set in `config.yaml` under the `sam3` section.

You need to run `login_helper.py` first to authenticate with HuggingFace before downloading model weights.

### yolo26_pipeline.py

Trains YOLOv26s with LOOCV. For each fold it extracts annotated frames from the training participants, converts bounding boxes from pixel coordinates to YOLO normalized format, splits 80/20 train/val, writes a `data.yaml`, trains for 100 epochs (configurable), then runs inference on the held-out participant. Same metrics as SAM 3 are computed.

### yolo26_sam3_combined.py

Same structure as above but instead of ground truth labels it reads SAM 3 prediction CSVs from the SAM 3 results directory and uses those as training labels. The SAM 3 bounding boxes are clamped to [0,1] range since masks can sometimes go slightly out of frame. Evaluation is still against ground truth so the comparison is fair.

### results_analysis.py

Runs after all three pipelines. Loads all the metrics and prediction CSVs and generates:
- Bar charts and boxplots comparing mAP@50 across methods and participants
- Welch's t-tests on mAP@50 between SAM 3 vs YOLO
- Motion metrics (path length in pixels, usage time in seconds) from predicted bounding box centers
- Mann-Whitney U tests comparing expert vs novice motion metrics
- Spearman correlations between motion metrics and ICO-OSCAR expert rubric scores, shown as heatmaps

The ICO-OSCAR CSV (`results_expert_evaluations_1.csv`) needs to be in the project root for the Spearman correlations. If it is missing, everything else still runs fine.

### utils.py

Shared utility module used by all pipelines. Contains config loading, dataset path construction, ground truth CSV parsing, bounding box format conversion (pixel to YOLO normalized), IoU calculation, COCO-style mAP computation, motion metric extraction, and CSV export helpers.

### login_helper.py

Logs you into HuggingFace. SAM 3 weights are gated so you need an account and access token. Only need to run this once; the token gets cached locally. You will need to request access to use the SAM 3 model from Hugging Face with your own API key to login. This is a quick script for personal use to bypass some terminal issues when copying the key.

## How to Run

**Requirements:** Python 3.10+, a CUDA GPU (SAM 3 needs significant VRAM), and access to the dataset.

### Environment Setup

The conda environment is defined in `environment.yml`. Solving can take a while depending on your system. Using the `libmamba` solver speeds it up significantly:

```bash
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

Then create and activate the environment:

```bash
conda env create -f environment.yml
conda activate cataract-tracking
```

If you are already on the **Tibia** computer in the lab, the `sam3_cataract` environment is already built and ready to use:

```bash
conda activate sam3_cataract
```

### HuggingFace Authentication

SAM 3 weights are gated on HuggingFace, so you need to log in once before running the SAM 3 pipeline:

```bash
python login_helper.py
```

### Running the Pipelines

Update `config.yaml` with your dataset paths and any settings you want to change, then run through `run.py`:

```bash
python run.py sam3
python run.py yolo
python run.py yolo-sam3
python run.py analyze
```

Or run everything in sequence:

```bash
python run.py all
```

SAM 3 takes the longest since it runs every frame through a large transformer. YOLO training is faster but still takes a while across 5 folds at 100 epochs each. Expect a few hours total on a single GPU.

## Configuration

All pipeline settings are centralized in `config.yaml`. Key sections:

- **dataset** - base path to the PerkTutor data, video/label subdirectories, FPS
- **participants / trials_per_participant** - which participants and trials to process
- **skill_map** - maps each participant to expert or novice for statistical tests
- **tool_classes** - the 6 instrument classes (must match ground truth CSV naming)
- **output** - where each pipeline saves its results
- **sam3** - model name, dtype, chunk size, tracking mode
- **yolo** - model weights path, epochs, confidence threshold, val split fraction, random seed
- **motion** - gap threshold for path length calculation
- **expert_scores_csv** - path to ICO-OSCAR evaluation CSV (set to `null` if unavailable)

See the comments in `config.yaml` for details on each setting.

## Tool Classes

6 surgical instruments tracked across all pipelines:
- capsulorhexis forceps
- cystotome needle
- diamond keratome iso
- diamond keratome straight
- forceps
- viscoelastic cannula

The `eye` and `lens` classes in the ground truth CSVs are filtered out automatically since we only track instruments.

## Metrics

- **mAP@50** - mean average precision at 0.50 IoU threshold
- **mAP@75** - mean average precision at 0.75 IoU threshold
- **mAP@50-95** - mean average precision averaged across 0.50 to 0.95 IoU (COCO-style)
- **Precision@50** and **Recall@50** at 0.50 IoU
- **F1@50** - harmonic mean of precision and recall
- **Path Length** - total Euclidean distance in pixels the bounding box center travels across frames (5 second gap threshold so detection dropouts do not inflate it)
- **Usage Time** - number of frames with a detection divided by FPS

## Dependencies

Defined in `environment.yml`. Main packages are PyTorch, HuggingFace Transformers (installed from git for SAM 3 support), Ultralytics for YOLO, OpenCV, pandas, scipy, seaborn, and matplotlib.

## Acknowledgements

- PerkTutor CataractSurgery dataset from the Perk Lab, Dr. Hisey, and everyone at the Perk Lab who helped me out! Thank you!
