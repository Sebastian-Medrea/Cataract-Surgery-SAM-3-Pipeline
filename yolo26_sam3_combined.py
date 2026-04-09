"""
YOLOv26 Trained on SAM 3 Zero-Shot Predictions for Cataract Surgery Instrument Detection.

CISC 499
Student: Sebastian Medrea
Supervisor: Dr. Rebecca Hisey

This program uses SAM 3 sequential zero-shot predictions as pseudo-labels to train a YOLOv26s model.
Instead of using ground truth bounding boxes for training, the SAM 3 predicted bounding boxes
serve as training labels. The trained YOLO model is then evaluated against ground truth using the
same LOOCV structure as the GT-trained YOLO version for a fair comparison.

Pipeline:
- Read SAM 3 sequential prediction CSVs per tool
- Extract video frames where SAM 3 made a detection
- Use SAM 3 predicted bounding boxes as YOLO training labels (pseudo-labels)
- Train YOLOv26s with same LOOCV as the GT-trained version
- Evaluate on held-out participant against ground truth

Inputs:
- Videos: .mkv files from the PerkTutor CataractSurgery dataset
- SAM 3 predictions: .csv files from the sequential SAM 3 pipeline results
- Labels: .csv files with ground truth bounding boxes (for evaluation only)

Outputs:
- predictions.csv: Frame-by-frame detection results per tool
- metrics.csv: Evaluations (mAP, Precision, Recall)
- visualizations/: PNG overlay of bounding boxes

Output structure:
  {output_dir}/Fold_P1/P1.1/P1.1_metrics.csv
  {output_dir}/Fold_P1/P1.1/P1.1_{tool}_predictions.csv
"""

import csv
import gc
import glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import cv2
import torch
import time
import yaml
import random

from ultralytics import YOLO, settings

# Disable Weights & Biases logging
os.environ["WANDB_DISABLED"] = "true"
settings.update({"wandb": False})

# Shared utilities
from utils import (
    load_config, DatasetConfig, get_tool_classes, get_class_mapping,
    load_annotations, get_video_fps, xyxy_to_yolo, parse_bbox,
    calculate_iou, calculate_metrics, export_predictions_csv,
    export_metrics_csv
)
 
# Load config for tool class mapping and SAM 3 results path
_config = load_config()
TOOL_CLASSES = get_tool_classes(_config)
CLASS_MAPPING, INV_CLASS_MAPPING = get_class_mapping(_config)
SAM3_RESULTS_DIR = _config['output']['sam3_results']
 
PARTICIPANTS = _config['participants']
TRIALS = {p: _config['trials_per_participant'] for p in PARTICIPANTS}
 
def sam3_pred_csv_path(participant, trial, tool):
    '''
    Build the path to a SAM 3 sequential per-tool prediction CSV. Tries the space-separated
    tool name first, then the underscore variant since the sequential naming is inconsistent
    between runs (e.g. 'capsulorhexis forceps' vs 'capsulorhexis_forceps').

    Args:
        participant (str): Participant ID (ex. 'P1')
        trial (int): Trial number (ex. 1)
        tool (str): Tool class name (ex. 'forceps')

    Returns:
        Path: Path to the prediction CSV (may not exist if SAM 3 was not run for this tool)
    '''

    folder = f"{participant}_{trial}"
    base = Path(SAM3_RESULTS_DIR) / participant / folder

    # Try with space first (simultaneous naming)
    space_path = base / f"{participant}_{trial}_{tool}_predictions.csv"
    
    if space_path.exists():
        return space_path

    # Try with underscore (sequential naming for capsulorhexis_forceps etc issue in dataset)
    underscore_tool = tool.replace(' ', '_')
    underscore_path = base / f"{participant}_{trial}_{underscore_tool}_predictions.csv"

    if underscore_path.exists():
        return underscore_path

    # Return the space version (will trigger the "not found" warning downstream)
    return space_path

def load_sam3_predictions_for_video(participant, trial):
    '''
    Load all SAM 3 sequential predictions for one video across all tools.
    Reads each per-tool prediction CSV and combines them into a single frame-indexed dict.

    Args:
        participant (str): Participant ID (ex. 'P1')
        trial (int): Trial number (ex. 1)

    Returns:
        dict: {frame_idx: [{'bbox': [x1, y1, x2, y2], 'class': tool_name}, ...]}
    '''

    frame_annotations = {}

    # Read each tool's prediction CSV and combine into one dict keyed by frame
    for tool in TOOL_CLASSES:
        csv_path = sam3_pred_csv_path(participant, trial, tool)

        if not csv_path.exists():

            print(f"Warning: {csv_path} not found")
            continue

        try:

            with open(csv_path, 'r') as f:
                for row in csv.DictReader(f):

                    # Parse the predicted bbox from the CSV string
                    pred_bbox = parse_bbox(row['pred_bbox'])
                    if pred_bbox is None:
                        continue

                    # Skip zero-confidence or negative-area predictions
                    score = float(row.get('pred_score', 0))
                    if score <= 0:
                        continue

                    if pred_bbox[2] <= pred_bbox[0] or pred_bbox[3] <= pred_bbox[1]:
                        continue

                    frame_idx = int(row['frame'])

                    if frame_idx not in frame_annotations:
                        frame_annotations[frame_idx] = []

                    frame_annotations[frame_idx].append({
                        'bbox': pred_bbox, 'class': tool
                    })

        except Exception as e:
            print(f"Error reading {csv_path}: {e}")

    return frame_annotations

def prepare_sam3_labeled_dataset(config, train_participants, output_dir, test_participant, val_fraction=0.2, seed=42):
    '''
    Extract video frames where SAM 3 made a detection and use the SAM 3 predicted bounding
    boxes as pseudo-labels in YOLO format. Same directory structure as the GT-trained version
    (train/images, train/labels, val/images, val/labels).

    Args:
        config (DatasetConfig): Dataset path config
        train_participants (list): Participants used for training/validation
        output_dir (str): Root output directory
        test_participant (str): Held-out participant for this fold
        val_fraction (float): Fraction of frames to use for validation
        seed (int): Random seed for reproducible train/val split

    Returns:
        Path: Path to the generated data.yaml for YOLO training
    '''

    # Setup YOLO directory structure
    fold_dir = Path(output_dir) / f"YOLO_SAM3_Dataset_Test_{test_participant}"
    train_img_dir = fold_dir / 'train' / 'images'
    train_lbl_dir = fold_dir / 'train' / 'labels'
    val_img_dir = fold_dir / 'val' / 'images'
    val_lbl_dir = fold_dir / 'val' / 'labels'

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    rng = random.Random(seed)
    total_train, total_val, total_skipped = 0, 0, 0

    for participant in train_participants:
        for trial in TRIALS[participant]:

            vid_path = config.get_video_path(participant, trial)

            if not vid_path.exists():
                print(f"Video not found: {vid_path}")
                continue

            # Load SAM 3 predictions instead of ground truth labels
            sam3_preds = load_sam3_predictions_for_video(participant, trial)
            if not sam3_preds:
                print(f"No SAM 3 predictions for {participant}.{trial}")
                continue

            annotated_frames = sorted(sam3_preds.keys())
            print(f"{participant}.{trial}: {len(annotated_frames)} frames with SAM 3 predictions")

            # Shuffle and split 80/20 for train/val at the frame level
            shuffled = annotated_frames[:]
            rng.shuffle(shuffled)

            split_idx = max(1, int(len(shuffled) * (1 - val_fraction)))
            train_frames = set(shuffled[:split_idx])

            # Read video and extract only frames where SAM 3 had a prediction
            cap = cv2.VideoCapture(str(vid_path))
            frame_idx = 0

            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                if frame_idx in sam3_preds:

                    is_val = frame_idx not in train_frames
                    img_dir = val_img_dir if is_val else train_img_dir
                    lbl_dir = val_lbl_dir if is_val else train_lbl_dir

                    base_name = f"{participant}_{trial}_{frame_idx:06d}"

                    # Convert SAM 3 predictions to YOLO label format
                    label_lines = []
                    for item in sam3_preds[frame_idx]:
                        tool_class = item['class']

                        if tool_class not in INV_CLASS_MAPPING:
                            continue

                        cls_id = INV_CLASS_MAPPING[tool_class]
                        xc, yc, w, h = xyxy_to_yolo(frame.shape, item['bbox'])

                        # Clamp to valid range since SAM 3 predictions can be slightly out of bounds
                        xc = max(0.0, min(1.0, xc))
                        yc = max(0.0, min(1.0, yc))
                        w = max(0.001, min(1.0, w))
                        h = max(0.001, min(1.0, h))

                        label_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

                    # Only save frame if there are valid labels
                    if label_lines:

                        cv2.imwrite(str(img_dir / f"{base_name}.jpg"), frame)
                        with open(lbl_dir / f"{base_name}.txt", 'w') as f:
                            f.write('\n'.join(label_lines) + '\n')

                        if is_val:
                            total_val += 1
                        else:
                            total_train += 1

                frame_idx += 1
            cap.release()

    print(f"  SAM3-labeled fold {test_participant}: {total_train} train, {total_val} val frames")

    # Write data.yaml that YOLO reads for training paths and class names
    yaml_path = fold_dir / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump({
            'train': str(train_img_dir.absolute()),
            'val': str(val_img_dir.absolute()),
            'names': CLASS_MAPPING
        }, f)

    return yaml_path


class YOLOTracker:
    '''
    Wrapper for YOLO model training and inference using the Ultralytics library.

    Model weights are loaded, trained on the fold's SAM 3 pseudo-labeled dataset, and
    then used to run inference on held-out test videos.
    '''

    def __init__(self, model_path="yolo26s.pt"):
        '''
        Init with pretrained YOLO weights.
        '''

        # Use NVIDIA GPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Device: {self.device}")

        self.model = YOLO(model_path)
        print(f"YOLO loaded: {model_path}")

    def train(self, data_yaml, epochs=100, project_dir="./results"):
        '''
        Train the YOLO model on the SAM 3 pseudo-labeled dataset for this fold.

        Args:
            data_yaml (str): Path to data.yaml config
            epochs (int): Number of training epochs
            project_dir (str): Directory to save training outputs

        Returns:
            Path: Path to the best model weights after training
        '''

        print(f"Training on {data_yaml}.")
        abs_project_dir = str(Path(project_dir).absolute())

        results = self.model.train(
            data=data_yaml, epochs=epochs, device=self.device,
            project=abs_project_dir, name="train",
            exist_ok=True, verbose=False
        )

        # Load the best weights from training for inference
        best_weights = Path(results.save_dir) / "weights" / "best.pt"
        print(f"Best weights: {best_weights}")
        self.model = YOLO(str(best_weights))

        return best_weights

    def track_video(self, video_path, tool_info_dict, class_mapping):
        '''
        Run YOLO inference on every frame of a video and collect per-tool predictions.

        Args:
            video_path (Path): Path to the video
            tool_info_dict (dict): Dictionary containing start/end frames for each tool
            class_mapping (dict): Map of class index to class name

        Returns:
            dict: Nested dictionary of predictions {tool_name: {frame_idx: [predictions]}}.
        '''

        all_predictions = {tool: {} for tool in tool_info_dict.keys()}

        cap = cv2.VideoCapture(str(video_path))
        frame_idx = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # Run YOLO on single frame
            conf = _config['yolo'].get('confidence_threshold', 0.25)
            results = self.model.predict(frame, verbose=False, conf=conf)
            boxes = results[0].boxes

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                score = float(boxes.conf[i].item())
                xyxy = boxes.xyxy[i].cpu().numpy()

                # Skip classes not in our mapping
                if cls_id not in class_mapping:
                    continue

                tool_class = class_mapping[cls_id]

                # Only store predictions within the tool's active frame window
                if tool_class in tool_info_dict:

                    first_frame = tool_info_dict[tool_class]['first_frame']
                    last_frame = tool_info_dict[tool_class]['last_frame']

                    if first_frame <= frame_idx <= last_frame:

                        if frame_idx not in all_predictions[tool_class]:
                            all_predictions[tool_class][frame_idx] = []

                        all_predictions[tool_class][frame_idx].append({
                            'bbox': [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                            'score': score, 'obj_id': 1
                        })

                        # Save original frame for visualization later
                        mask_dir = tool_info_dict[tool_class]['mask_dir']
                        os.makedirs(mask_dir, exist_ok=True)
                        frame_path = f'{mask_dir}/frame_{frame_idx:04d}.npy'

                        if not os.path.exists(frame_path):
                            np.save(frame_path, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            frame_idx += 1

        cap.release()
        return all_predictions

def visualize_tool_boxes(tool_name, annotations, predictions, mask_dir, output_dir):
    '''
    Generates PNG images comparing Ground Truth vs. YOLO Prediction (SAM 3-trained).
    
    Args:
        tool_name (str): Name of the tool ('Forceps')
        annotations (dict): Ground truth data
        predictions (dict): Model predictions
        mask_dir (Path): Source directory of saved numpy frames
        output_dir (Path): Destination directory for PNGs
    '''

    # Find all saved frame numpy files
    frame_files = sorted(glob.glob(f'{mask_dir}/frame_*.npy'))
    if not frame_files:
        return

    print(f"Visualizing {len(frame_files)} boxes for {tool_name}.")

    for frame_file in frame_files:

        # Grab frame number from file name
        frame_num = int(Path(frame_file).stem.split('_')[-1])

        # Load original image frame
        frame = np.load(frame_file)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Ground truth
        axes[0].imshow(frame)

        if frame_num in annotations:
            for item in annotations[frame_num]:

                bbox = item['bbox']
                axes[0].add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                                  fill=False, edgecolor='lime', linewidth=2))

                # Draw class label above bbox
                axes[0].text(bbox[0], max(0, bbox[1]-8), item['class'], color='lime',
                             fontsize=10, fontweight='bold',
                             bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=2))

        axes[0].set_title(f'Frame {frame_num} - Ground Truth')
        axes[0].axis('off')

        # Prediction
        axes[1].imshow(frame)
        if frame_num in predictions and predictions[frame_num]:
            bbox = predictions[frame_num][0]['bbox']
            axes[1].add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                              fill=False, edgecolor='cyan', linewidth=2))

            # Draw predicted class label above bbox
            axes[1].text(bbox[0], max(0, bbox[1]-8), tool_name, color='cyan',
                         fontsize=10, fontweight='bold',
                         bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=2))

            if frame_num in annotations:
                for gt_item in annotations[frame_num]:
                    if gt_item['class'] == tool_name:
                        gt = gt_item['bbox']
                        iou = calculate_iou(gt, bbox)

                        # IoU text at bottom left to avoid overlapping tool label
                        axes[1].text(10, frame.shape[0]-30, f'IoU: {iou:.3f}', color='white',
                                   fontsize=12, bbox=dict(facecolor='green' if iou > 0.5 else 'red', alpha=0.8))
                        break

        axes[1].set_title('YOLO Prediction (SAM3-trained)')
        axes[1].axis('off')

        plt.tight_layout()
        plt.savefig(output_dir / f'{tool_name}_frame{frame_num:04d}.png', dpi=100)
        plt.close()


def process_sam3_to_yolo(participants=None, output_dir='./yolo26_sam3_trained', model_path='yolo26s.pt', epochs=100):
    '''
    Run YOLO LOOCV pipeline trained on SAM 3 pseudo-labels. Each fold holds out one participant
    for testing against ground truth and trains on SAM 3 predictions from the remaining four.

    Args:
        participants (list): List like ['P1', 'P2'] or None for all
        output_dir (str): Where to save results
        model_path (str): Path to pretrained YOLO weights
    '''

    if participants is None:
        participants = PARTICIPANTS

    config = DatasetConfig(_config)
    all_results = {}

    # LOOCV: each participant takes a turn as the held-out test set
    for test_participant in participants:
        print(f"LOOCV Fold: Test = {test_participant}")
        print(f"Training YOLO on SAM 3 sequential pseudo-labels")
        

        fold_dir = Path(output_dir) / f"Fold_{test_participant}"
        os.makedirs(fold_dir, exist_ok=True)

        # Everyone except the test participant trains on SAM 3 labels
        train_participants = [p for p in participants if p != test_participant]
        print(f"Train (SAM3 labels): {train_participants} | Test (GT eval): {test_participant}")

        # Prepare dataset using SAM 3 pseudo-labels and train
        yaml_path = prepare_sam3_labeled_dataset(
            config, train_participants, output_dir, test_participant
        )

        tracker = YOLOTracker(model_path)
        tracker.train(str(yaml_path), epochs=epochs, project_dir=str(fold_dir))

        # Evaluate the SAM3-trained model on each trial of the held-out test participant
        for trial in TRIALS[test_participant]:
            dataset_key = f"{test_participant}.{trial}"
            print(f"\nEvaluating: {dataset_key} (against GT)")

            video_path = config.get_video_path(test_participant, trial)
            csv_path = config.get_csv_path(test_participant, trial)

            if not video_path.exists() or not csv_path.exists():
                print(f"Files not found")
                continue

            # Setup folders
            trial_dir = fold_dir / dataset_key
            masks_dir = trial_dir / 'masks'
            vis_dir = trial_dir / 'visualizations'

            for d in [trial_dir, masks_dir, vis_dir]:
                os.makedirs(d, exist_ok=True)

            # Load ground truth annotations for evaluation
            fps = get_video_fps(video_path)
            gt_annotations = load_annotations(csv_path, fps)

            # Find active tools
            all_tools = set()

            for frame_data in gt_annotations.values():
                for item in frame_data:

                    all_tools.add(item['class'])

            # Build tool info
            tool_info_dict = {}

            for tool_class in sorted(all_tools):
                tool_frames = sorted([
                    f for f, items in gt_annotations.items()
                    if any(item['class'] == tool_class for item in items)
                ])

                if tool_frames:
                    tool_info_dict[tool_class] = {
                        'first_frame': tool_frames[0],
                        'last_frame': tool_frames[-1],
                        'mask_dir': masks_dir / tool_class.replace(' ', '_')
                    }

            # Run inference
            start_time = time.time()
            all_tool_predictions = tracker.track_video(video_path, tool_info_dict, CLASS_MAPPING)
            elapsed = time.time() - start_time

            # Calculate metrics per tool
            all_predictions = {}
            all_metrics = {}

            for tool_class in sorted(all_tools):
                tool_predictions = all_tool_predictions.get(tool_class, {})

                # Merge into combined predictions dict (all tools, all frames)
                for frame_idx, preds in tool_predictions.items():
                    for pred in preds:
                        pred['tool_class'] = tool_class

                    if frame_idx not in all_predictions:
                        all_predictions[frame_idx] = []

                    all_predictions[frame_idx].extend(preds)

                # Filter GT to this tool for per-tool metrics
                tool_annotations = {
                    f: [item for item in items if item['class'] == tool_class]
                    for f, items in gt_annotations.items()
                    if any(item['class'] == tool_class for item in items)
                }

                all_metrics[tool_class] = calculate_metrics(tool_annotations, tool_predictions)

                export_predictions_csv(
                    tool_predictions, gt_annotations, fps,
                    trial_dir / f"{dataset_key}_{tool_class}_predictions.csv", tool_class
                )

                # Visualize
                if tool_class in tool_info_dict:
                    visualize_tool_boxes(
                        tool_class, tool_annotations, tool_predictions,
                        tool_info_dict[tool_class]['mask_dir'], vis_dir
                    )

            # Compute overall metrics across all tools combined
            overall_metrics = calculate_metrics(gt_annotations, all_predictions)
            all_metrics['OVERALL'] = overall_metrics
            export_metrics_csv(all_metrics, trial_dir / f"{dataset_key}_metrics.csv")

            print(f"mAP@50={overall_metrics['mAP@50']:.4f} F1={overall_metrics['f1@50']:.4f} time={elapsed:.1f}s")
            all_results[dataset_key] = {'metrics': all_metrics, 'time': elapsed}

            torch.cuda.empty_cache()
            gc.collect()

    # Write summary CSV across all folds
    print("ALL DATASETS SUMMARY")

    summary_rows = []
    for dataset_key, result in all_results.items():

        overall = result['metrics'].get('OVERALL', {})
        summary_rows.append({
            'dataset': dataset_key,
            'time_sec': f"{result['time']:.1f}",
            'mAP@50': f"{overall.get('mAP@50', 0):.4f}",
            'mAP@75': f"{overall.get('mAP@75', 0):.4f}",
            'mAP@50-95': f"{overall.get('mAP@50-95', 0):.4f}",
            'f1': f"{overall.get('f1@50', 0):.4f}"
        })

    summary_csv = Path(output_dir) / 'yolo_sam3_summary_metrics.csv'

    with open(summary_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['dataset', 'time_sec', 'mAP@50',
                                               'mAP@75', 'mAP@50-95', 'f1'])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nDone. Results in {output_dir}")


def main():
    '''
    Main control flow; runs LOOCV training YOLO on SAM 3 pseudo-labels.
    '''

    MODEL_PATH = 'yolo26s.pt'
    OUTPUT_DIR = './yolo26_sam3_trained'

    process_sam3_to_yolo(
        participants=['P1', 'P2', 'P3', 'P4', 'P5'],
        output_dir=OUTPUT_DIR,
        model_path=MODEL_PATH
    )

    print(f"Training complete.")


if __name__ == '__main__':
    main()
