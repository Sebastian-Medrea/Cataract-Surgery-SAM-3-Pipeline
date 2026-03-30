# Cataract Surgery Tool Tracking: SAM 3 vs YOLOv26

CISC 499 Project

Sebastian Medrea, supervised by Dr. Rebecca Hisey

## About

This project compares two different deep learning methods for automatically tracking surgical instruments in simulated cataract surgery videos. Basically, can we streamline a method for skill assessment that provides a cheaper and less invasive alternative to EM tracking, and can these models automate some part of the annotation process to reduce the burden for expert labelling?

The two main methods are:

1. **SAM 3** (Segment Anything Model 3) from Meta. This is a zero-shot approach where I just give the model one bounding box on the first frame a tool shows up, and it tracks the tool through the rest of the video on its own. No training needed.

2. **YOLOv26** trained with Leave-One-Out Cross Validation. Each fold holds out one participant for testing and trains on the other four. This is the standard supervised method that needs actual ground truth labels.

There is also a third experiment where I train YOLO using SAM 3's predictions as pseudo-labels instead of ground truth. The point is to see if SAM 3 can generate usable training data on its own.

All three are evaluated against ground truth bounding boxes using mAP, precision, recall, and F1. I also compute motion metrics (path length and usage time) from the predicted bounding boxes and run statistical tests to see if the tracking can tell apart expert surgeons from novices.

## Dataset

Data comes from the **PerkTutor CataractSurgery** simulated dataset on the Perk Lab network drive:

```
P:/data/PerkTutor/CataractSurgery/
    Videos/Simulated_Data/mkv/
    Datasets/Simulated_Data/
```

5 participants (P1 through P5), 5 trials each, so 25 videos total at 15 FPS. P1 is the expert ophthalmologist and P2 through P5 are novice residents. Ground truth CSVs have timestamped bounding boxes for 6 tool classes. P1 has revised labels so there is separate path handling for that in the code.

The dataset is not in this repo. You need access to the Perk Lab drive or a local copy. Change `base_path` in `DatasetConfig` if needed.

## Files

- `sam3_pipeline.py` - SAM 3 zero-shot tracking pipeline
- `yolo26_pipeline.py` - YOLOv26 training and inference with ground truth labels
- `yolo26_sam3_combined.py` - YOLOv26 trained on SAM 3 pseudo-labels
- `results_analysis.py` - statistical analysis and all the plots
- `login_helper.py` - logs you into HuggingFace for SAM 3 model weights
- `requirements.txt` - dependencies

## Scripts

### sam3_pipeline.py

Loads the SAM 3 model from HuggingFace in bfloat16 and runs zero-shot bounding box tracking on the cataract videos. It reads ground truth CSVs to find where each tool first appears, gives SAM 3 that initial bounding box as a prompt, and then SAM 3 propagates through the video in chunks of 100 frames. The output masks get converted back to bounding boxes and compared against ground truth.

Two tracking modes:
- `track_sequential` tracks one tool at a time, resets the model memory between tools
- `track_simultaneous` tracks all tools in one pass with multi-object tracking

Set `MODE` and `RUN_SINGLE` in `main()` to control which mode and whether to run one video or all 25.

You need to run `login_helper.py` first to authenticate with HuggingFace before downloading model weights.

### yolo26_pipeline.py

Trains YOLOv26s with LOOCV. For each fold it extracts annotated frames from the training participants, converts bounding boxes from pixel coordinates to YOLO normalized format, splits 80/20 train/val, writes a `data.yaml`, trains for 100 epochs, then runs inference on the held-out participant. Same metrics as SAM 3 are computed.

### yolo26_sam3_combined.py

Same thing as above but instead of ground truth labels it reads SAM 3 prediction CSVs from `./results/` and uses those as training labels. The SAM 3 bounding boxes are clamped to [0,1] range since masks can sometimes go slightly out of frame. Evaluation is still against ground truth so the comparison is fair.

### results_analysis.py

Runs after all three pipelines. Loads all the metrics and prediction CSVs and generates:
- Bar charts and boxplots comparing mAP@50 across methods and participants
- Welch's t-tests on mAP@50 between SAM 3 vs YOLO
- Motion metrics (path length in pixels, usage time in seconds) from predicted bounding box centers
- Mann-Whitney U tests comparing expert vs novice motion metrics
- Spearman correlations between motion metrics and ICO-OSCAR expert rubric scores, shown as heatmaps

The ICO-OSCAR CSV (`results_expert_evaluations_1.csv`) needs to be in the project root for the Spearman part. If its missing, everything else still runs fine.

### login_helper.py

Just logs you into HuggingFace. SAM 3 weights are gated so you need an account and token. Only need to run this once, the token gets cached.

## How to Run

**You need:** Python 3.10+, a CUDA GPU (SAM 3 needs a lot of VRAM), and access to the dataset.

```bash
cd Cataract-Surgery-SAM-3-Pipeline
pip install -r requirements.txt
python login_helper.py
```

Then run in order since each script depends on the previous output:

```bash
python sam3_pipeline.py
python yolo26_pipeline.py
python yolo26_sam3_combined.py
python results_analysis.py
```

SAM 3 takes the longest since it runs every frame through a big transformer. YOLO training is faster but still takes a bit across 5 folds at 100 epochs each. Expect a few hours total.

## Configuration

Each script has globals in `main()` or at the top of the file:
- `sam3_pipeline.py`: `MODE` for sequential/simultaneous, `RUN_SINGLE` for one video or all
- `yolo26_pipeline.py` / `yolo26_sam3_combined.py`: participants, trials, model path, output dir are set in `main()`
- `results_analysis.py`: `SAM3_DIR`, `YOLO_DIR`, `OUTPUT_DIR`, `EXPERT_SCORES_CSV` at the top
- All pipelines: `DatasetConfig` defaults to `P:/data/PerkTutor/CataractSurgery`, change `base_path` if needed

## Tool Classes

6 surgical instruments tracked across all pipelines:
- capsulorhexis forceps
- cystotome needle
- diamond keratome iso
- diamond keratome straight
- forceps
- viscoelastic cannula

The `eye` class in the ground truth CSVs is filtered out since we only care about instruments.

## Metrics

- **mAP@50** - mean average precision at 0.50 IoU
- **mAP@75** - mean average precision at 0.75 IoU
- **mAP@50-95** - mean average precision averaged from 0.50 to 0.95 IoU
- **Precision@50** and **Recall@50** at 0.50 IoU
- **F1@50** - harmonic mean of precision and recall
- **Path Length** - total distance in pixels the bounding box center moves across frames (5 second gap threshold so detection dropouts dont inflate it)
- **Usage Time** - frames with a detection / FPS

## Dependencies

Everything is in `requirements.txt`. Main ones are PyTorch, HuggingFace Transformers (from git for SAM 3), Ultralytics for YOLO, OpenCV, pandas, scipy, seaborn, and matplotlib.

## Acknowledgements

- PerkTutor CataractSurgery dataset from the Perk Lab, Dr. Hisey, and everyone at the Perk Lab who helped me out! Thank you!
