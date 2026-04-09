"""
YOLOv26 Training on Surgical Tool Object Detection for Cataract Surgery.

CISC 499
Student: Sebastian Medrea
Supervisor: Dr. Rebecca Hisey

This program uses YOLOv26s (small) to train and evaluate surgical tool detection using
Leave-One-Out Cross Validation (LOOCV) on the PerkTutor CataractSurgery dataset. Each fold
holds out one participant for testing while the remaining four are split 80/20 at the frame level
for training and validation.

mAP50 metrics, IoU, amongst others are compared against ground truth bounding boxes and the 
predicted YOLO bounding boxes.

Inputs:
- Videos: .mkv files from the PerkTutor CataractSurgery dataset
- Labels: .csv files with ground truth bounding boxes

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

from utils import (
    load_config, DatasetConfig, get_tool_classes, get_class_mapping,
    load_annotations, get_video_fps, xyxy_to_yolo,
    calculate_iou,
    calculate_metrics, 
    export_predictions_csv, export_metrics_csv
)

# Load config for tool class mapping
_config = load_config()
TOOL_CLASSES = get_tool_classes(_config)
CLASS_MAPPING, _ = get_class_mapping(_config)


def prepare_yolo_dataset(config, train_participants, trials, output_dir, class_mapping, test_participant, val_fraction=0.2, seed=42):
    '''
    Extract video frames and their bounding box labels into YOLO's directory
    structure (train/images, train/labels, val/images, val/labels) with a data.yaml config.

    Args:
        config (DatasetConfig): Dataset path config
        train_participants (list): Participants used for training/validation
        trials (dict): Trial numbers per participant
        output_dir (str): Root output directory
        class_mapping (dict): Map of class index to class name
        test_participant (str): Held-out participant for this fold
        val_fraction (float): Fraction of frames to use for validation
        seed (int): Random seed for reproducible train/val split

    Returns:
        Path: Path to the generated data.yaml for YOLO training
    '''

    # Setup YOLO directory structure
    fold_dir = Path(output_dir) / f"YOLO_Dataset_Test_{test_participant}"
    train_img_dir = fold_dir / 'train' / 'images'
    train_lbl_dir = fold_dir / 'train' / 'labels'
    val_img_dir = fold_dir / 'val' / 'images'
    val_lbl_dir = fold_dir / 'val' / 'labels'

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    # Reverse map: class name to class index for writing YOLO label files
    inv_class_mapping = {v: k for k, v in class_mapping.items()}
    rng = random.Random(seed)
    total_train, total_val = 0, 0

    for participant in train_participants:
        for trial in trials[participant]:

            vid_path = config.get_video_path(participant, trial)
            csv_path = config.get_csv_path(participant, trial)

            if not vid_path.exists() or not csv_path.exists():
                continue

            fps = get_video_fps(vid_path)
            annotations = load_annotations(csv_path, fps)
            annotated_frames = sorted(annotations.keys())

            if not annotated_frames:
                continue

            # Shuffle and split 80/20 for train/val at the frame level
            shuffled = annotated_frames[:]
            rng.shuffle(shuffled)

            split_idx = max(1, int(len(shuffled) * (1 - val_fraction)))
            train_frames = set(shuffled[:split_idx])

            # Read video and extract only annotated frames
            cap = cv2.VideoCapture(str(vid_path))
            frame_idx = 0

            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                if frame_idx in annotations:

                    is_val = frame_idx not in train_frames
                    img_dir = val_img_dir if is_val else train_img_dir
                    lbl_dir = val_lbl_dir if is_val else train_lbl_dir

                    # Save frame as jpg and write YOLO-format label file
                    base_name = f"{participant}_{trial}_{frame_idx:06d}"
                    cv2.imwrite(str(img_dir / f"{base_name}.jpg"), frame)

                    with open(lbl_dir / f"{base_name}.txt", 'w') as f:

                        for item in annotations[frame_idx]:
                            if item['class'] in inv_class_mapping:

                                cls_id = inv_class_mapping[item['class']]
                                xc, yc, w, h = xyxy_to_yolo(frame.shape, item['bbox'])
                                f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

                    if is_val:
                        total_val += 1

                    else:
                        total_train += 1

                frame_idx += 1

            cap.release()

    print(f"Fold {test_participant}: {total_train} train, {total_val} val frames")

    # Write data.yaml that YOLO reads for training paths and class names
    yaml_path = fold_dir / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump({
            'train': str(train_img_dir.absolute()),
            'val': str(val_img_dir.absolute()),
            'names': class_mapping
        }, f)

    return yaml_path


class YOLOTracker:
    '''
    Wrapper for YOLO model training and inference using the Ultralytics library.

    Model weights are loaded, trained on the fold's dataset, and then used to run
    inference on held-out test videos.
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
        Train the YOLO model on the prepared dataset for this fold.

        Args:
            data_yaml (str): Path to data.yaml config
            epochs (int): Number of training epochs
            project_dir (str): Directory to save training outputs

        Returns:
            Path: Path to the best model weights after training
        '''

        print(f"Training on {data_yaml}...")
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
            
            conf = _config['yolo'].get('confidence_threshold', 0.25)

            # Run YOLO on single frame and extract detected boxes
            results = self.model.predict(frame, verbose=False, conf=conf)
            boxes = results[0].boxes

            # Iterate over each detected box in this frame
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
    Generates PNG images comparing Ground Truth vs. YOLO Prediction.
    
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

        axes[1].set_title('YOLO Prediction')
        axes[1].axis('off')

        plt.tight_layout()
        plt.savefig(output_dir / f'{tool_name}_frame{frame_num:04d}.png', dpi=100)
        plt.close()


def process_datasets_yolo(participants=None, trials=None, output_dir='./yolo26_results_fixed', model_path='yolo26s.pt', epochs=100):
    '''
    Run YOLO LOOCV pipeline across all participants. Each fold holds out one participant
    for testing and trains on the remaining four with 80/20 frame-level split.

    Args:
        participants (list): List like ['P1', 'P2'] or None for all
        trials (dict): Dict like {'P1': [1, 2], 'P2': [1]} or None for [1,2,3,4,5]
        output_dir (str): Where to save results
        model_path (str): Path to pretrained YOLO weights
    '''

    if participants is None:
        participants = ['P1', 'P2', 'P3', 'P4', 'P5']

    if trials is None:
        trials = {p: [1, 2, 3, 4, 5] for p in participants}

    config = DatasetConfig(_config)
    all_results = {}

    # LOOCV: each participant takes a turn as the held-out test set
    for test_participant in participants:
        print(f"LOOCV Fold: Test = {test_participant}")

        fold_dir = Path(output_dir) / f"Fold_{test_participant}"
        os.makedirs(fold_dir, exist_ok=True)

        # Everyone except the test participant trains
        train_participants = [p for p in participants if p != test_participant]
        print(f"Train/Val: {train_participants} (80/20 frame split); Test: {test_participant}")

        # Prepare YOLO dataset for this fold (extract frames + labels) and train
        yaml_path = prepare_yolo_dataset(config, train_participants, trials, output_dir,
                                         CLASS_MAPPING, test_participant)

        tracker = YOLOTracker(model_path)
        tracker.train(str(yaml_path), epochs=epochs, project_dir=str(fold_dir))

        # Evaluate on held-out test participant
        for trial in trials[test_participant]:

            dataset_key = f"{test_participant}.{trial}"
            print(f"\nEvaluating: {dataset_key}")

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

            # Load annotations
            fps = get_video_fps(video_path)
            annotations = load_annotations(csv_path, fps)

            # Find active tools
            all_tools = set()

            for frame_data in annotations.values():
                for item in frame_data:

                    all_tools.add(item['class'])

            # Build tool info: find first/last frame per tool for the active detection window
            tool_info_dict = {}
            for tool_class in sorted(all_tools):

                tool_frames = sorted([
                    f for f, items in annotations.items()
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

            # Evaluate each tool separately then combine for OVERALL
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

                # Filter GT annotations to only this tool for per-tool metrics
                tool_annotations = {
                    f: [item for item in items if item['class'] == tool_class]
                    for f, items in annotations.items()
                    if any(item['class'] == tool_class for item in items)
                }

                all_metrics[tool_class] = calculate_metrics(tool_annotations, tool_predictions)

                # Export CSV
                export_predictions_csv(
                    tool_predictions, annotations, fps,
                    trial_dir / f"{dataset_key}_{tool_class}_predictions.csv", tool_class
                )

                # Visualize
                visualize_tool_boxes(
                    tool_class, tool_annotations, tool_predictions,
                    tool_info_dict[tool_class]['mask_dir'], vis_dir
                )

            # Compute overall metrics across all tools combined
            overall_metrics = calculate_metrics(annotations, all_predictions)
            all_metrics['OVERALL'] = overall_metrics
            export_metrics_csv(all_metrics, trial_dir / f"{dataset_key}_metrics.csv")

            print(f"mAP@50={overall_metrics['mAP@50']:.4f} F1={overall_metrics['f1@50']:.4f} time={elapsed:.1f}s")
            all_results[dataset_key] = {'metrics': all_metrics, 'time': elapsed}

            # Free GPU memory before next trial
            torch.cuda.empty_cache()
            gc.collect()

    # Write summary CSV across all folds with one row per video
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

    summary_csv = Path(output_dir) / 'yolo_summary_metrics.csv'

    with open(summary_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['dataset', 'time_sec', 'mAP@50',
                                               'mAP@75', 'mAP@50-95', 'f1'])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nDone. Results in {output_dir}")


def main():
    '''
    Main control flow; runs LOOCV across all participants.
    '''

    # yolo26s.pt for small model
    MODEL_PATH = 'yolo26s.pt'
    OUTPUT_DIR = './yolo26_results_fixed'

    process_datasets_yolo(
        participants=['P1', 'P2', 'P3', 'P4', 'P5'],
        output_dir=OUTPUT_DIR,
        model_path=MODEL_PATH
    )

    print(f"Inference complete.")


if __name__ == '__main__':
    main()