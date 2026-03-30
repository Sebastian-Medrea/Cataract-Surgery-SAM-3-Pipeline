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

import ast
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

# Tool classes tracked in the dataset (no 'eye' or 'lens')
TOOL_CLASSES = sorted([
    'capsulorhexis forceps', 'cystotome needle', 'diamond keratome iso',
    'diamond keratome straight', 'forceps', 'viscoelastic cannula'
])

# Map class index to class name for YOLO format
CLASS_MAPPING = {i: name for i, name in enumerate(TOOL_CLASSES)}


class DatasetConfig:
    '''
    Handles path differences between P1-P5 datasets for the Perk Lab dataset structure with Revised and
    normal labels.
    '''

    def __init__(self, base_path="P:/data/PerkTutor/CataractSurgery"):
        '''
        Init with root directory.
        '''
        self.base_path = Path(base_path)

        # mkv files
        self.video_dir = self.base_path / "Videos/Simulated_Data/mkv"
        self.dataset_dir = self.base_path / "Datasets/Simulated_Data"

    def get_video_path(self, participant, trial):
        '''
        Construct path to video file.

        Args:
            participant (str): Participant ID (ex. 'P1')
            trial (int): Trial number (ex. 1)
        
        Returns:
            Pathlib object for .mkv
        '''

        # P1.1-1 naming
        video_name = f"{participant}.{trial}-1.mkv"
        return self.video_dir / video_name

    def get_csv_path(self, participant, trial):
        '''
        Construct path to CSV ground truth bounding box labels for each trial.
        '''

        # Only P1 has revised labels
        if participant == 'P1':
            folder = self.dataset_dir / f"P1_{trial}_Revised"
            csv_name = f"P1_{trial}_Revised_Labels.csv"

        else:
            folder = self.dataset_dir / f"{participant}_{trial}"
            csv_name = f"{participant}_{trial}_Labels.csv"

        return folder / csv_name


def xyxy_to_yolo(img_size, bbox):
    '''
    Convert bounding box from [xmin, ymin, xmax, ymax] pixel coords to YOLO normalized
    format [x_center, y_center, width, height] where values are 0-1.

    Args:
        img_size (tuple): Image shape (height, width, channels)
        bbox (list): [xmin, ymin, xmax, ymax]

    Returns:
        tuple: (x_center, y_center, width, height) normalized by image dimensions
    '''

    # Normalize by image width and height
    dw = 1.0 / img_size[1]
    dh = 1.0 / img_size[0]

    x_center = (bbox[0] + bbox[2]) / 2.0 * dw
    y_center = (bbox[1] + bbox[3]) / 2.0 * dh
    w = (bbox[2] - bbox[0]) * dw
    h = (bbox[3] - bbox[1]) * dh

    return x_center, y_center, w, h


def load_annotations(csv_path, fps):
    '''
    Grabs ground truth CSV files for bounding boxes from dataset.

    Args:
        csv_path (Path): Path to .csv file
        fps (float): Frames per second of video to convert time to frame index
        
    Returns:
        dict: A dictionary mapping frame_index to list of tool objects 
              {'bbox': [...], 'class': '...'}
    '''

    annotations = {}
    with open(csv_path, 'r') as f:
        for row in csv.DictReader(f):

            # CSV is logged in time, so multiply by FPS to get frame
            frame_idx = int(round(float(row['Time Recorded']) * fps))
            bbox_str = row['Tool bounding box']

            # No tools in this frame
            if bbox_str == '[]':
                continue

            # Convert lists stored as strings from the CSV to lists
            bboxes = ast.literal_eval(bbox_str)
            frame_tools = []

            for bbox in bboxes:

                # Ignore any class labelled eye and only track tools
                if bbox['class'].lower() != 'eye':
                    frame_tools.append({
                        'bbox': [bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']],
                        'class': (bbox['class']).lower().replace('_', ' ').strip()
                    })

            if frame_tools:
                annotations[frame_idx] = frame_tools

    return annotations


def get_video_fps(video_path):
    '''
    Grab FPS from video.
    '''
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    return fps


def prepare_yolo_dataset(config, train_participants, trials, output_dir,
                         class_mapping, test_participant, val_fraction=0.2, seed=42):
    '''
    Extract video frames and their bounding box labels into YOLO's expected directory
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

            # Run YOLO on single frame
            results = self.model.predict(frame, verbose=False, conf=0.25)
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


def calculate_iou(bbox1, bbox2):
    '''
    Calculate intersection over union between ground truth and predicted bounding box (area of overlap / area of union).

    Args:
        bbox1 (list): [xmin, ymin, xmax, ymax]
        bbox2 (list): [xmin, ymin, xmax, ymax]
        
    Returns:
        float: The IoU score (0.0 to 1.0)
    '''

    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    # Coordinates of intersection rectangle
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    # Area of intersection (max with 0 for no overlap)
    inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)

    # Area of both boxes
    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)

    # Union = Box Areas - Intersection
    union_area = bbox1_area + bbox2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def calculate_metrics(ground_truth, predictions):
    '''
    Computes object detection metrics for tools using dataset-wide AP (not per-frame).
    Calculates mAP (Mean Average Precision) at various IoU thresholds (0.5 to 0.95).
    
    Args:
        ground_truth (dict): Dictionary of ground truth annotations
        predictions (dict): Dictionary of model predictions
        
    Returns:
        dict: Dictionary containing 'mAP@50', 'mAP@75', 'f1', etc.
    '''

    iou_thresholds = np.linspace(0.5, 0.95, 10)

    # Evaluate over all ground truth frames
    eval_frames = sorted(ground_truth.keys())

    if not eval_frames:
        return {
            'mAP@50': 0.0, 'mAP@75': 0.0, 'mAP@50-95': 0.0,
            'precision@50': 0.0, 'recall@50': 0.0, 'f1@50': 0.0,
            'num_frames': 0
        }

    aps_per_threshold = {}

    for iou_thresh in iou_thresholds:

        # Collect all predictions across frames with their scores for ranking
        all_preds = []
        total_gt_boxes = 0

        for frame_idx in eval_frames:
            gt_bboxes = [item['bbox'] for item in ground_truth.get(frame_idx, [])]
            total_gt_boxes += len(gt_bboxes)

            for pred in predictions.get(frame_idx, []):
                all_preds.append({
                    'frame': frame_idx,
                    'bbox': pred['bbox'],
                    'score': pred['score']
                })

        # Sort predictions by confidence score for highest first
        all_preds = sorted(all_preds, key=lambda x: x['score'], reverse=True)
        true_positives = np.zeros(len(all_preds))
        false_positives = np.zeros(len(all_preds))
        matched_gts = {f: set() for f in eval_frames}

        for pred_idx, pred in enumerate(all_preds):
            frame_idx = pred['frame']
            gt_bboxes = [item['bbox'] for item in ground_truth.get(frame_idx, [])]
            max_iou = 0
            max_gt_idx = -1

            # Check prediction against all ground truth boxes
            for gt_idx, gt_bbox in enumerate(gt_bboxes):

                # Already matched
                if gt_idx in matched_gts[frame_idx]:
                    continue

                iou = calculate_iou(pred['bbox'], gt_bbox)
                if iou > max_iou:
                    max_iou = iou
                    max_gt_idx = gt_idx

            # Target hit
            if max_iou >= iou_thresh:
                true_positives[pred_idx] = 1
                matched_gts[frame_idx].add(max_gt_idx)
            else:
                false_positives[pred_idx] = 1

        # Calculate Average Precision using precision-recall curve
        if total_gt_boxes == 0:
            ap = 0.0
        else:
            tp_cumsum = np.cumsum(true_positives)
            fp_cumsum = np.cumsum(false_positives)
            recalls = tp_cumsum / total_gt_boxes
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum + np.finfo(float).eps)

            # Pad with boundary values for interpolation
            precisions = np.concatenate(([0.0], precisions, [0.0]))
            recalls = np.concatenate(([0.0], recalls, [1.0]))

            # Monotonically decreasing precision (standard AP interpolation)
            for i in range(len(precisions) - 1, 0, -1):
                precisions[i - 1] = max(precisions[i - 1], precisions[i])

            # Sum area under the precision-recall curve at recall change points
            indices = np.where(recalls[1:] != recalls[:-1])[0] + 1
            ap = sum((recalls[i] - recalls[i - 1]) * precisions[i] for i in indices)

        aps_per_threshold[round(iou_thresh, 2)] = ap

    # Precision/Recall/F1 at IoU=0.5
    total_tp = total_fp = total_fn = 0

    for frame_idx in eval_frames:

        gt_bboxes = [item['bbox'] for item in ground_truth.get(frame_idx, [])]
        pred_items = predictions.get(frame_idx, [])

        if not pred_items:
            total_fn += len(gt_bboxes)
            continue

        matched_gt = set()

        for pred in pred_items:
            matched = False

            for gt_idx, gt_bbox in enumerate(gt_bboxes):
                if gt_idx not in matched_gt:

                    if calculate_iou(pred['bbox'], gt_bbox) >= 0.5:
                        total_tp += 1
                        matched_gt.add(gt_idx)
                        matched = True
                        break

            if not matched:
                total_fp += 1

        total_fn += len(gt_bboxes) - len(matched_gt)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'mAP@50': float(aps_per_threshold[0.5]),
        'mAP@75': float(aps_per_threshold[0.75]),
        'mAP@50-95': float(np.mean(list(aps_per_threshold.values()))),
        'precision@50': float(precision),
        'recall@50': float(recall),
        'f1@50': float(f1),
        'num_frames': len(eval_frames)
    }


def export_predictions_csv(predictions, annotations, fps, output_path, tool_class):
    '''
    Writes the frame-by-frame detection results to CSV.
    
    Args:
        predictions (dict): Model predictions
        annotations (dict): Ground truth data (for comparison)
        fps (float): Video FPS
        output_path (Path): Destination file path
        tool_class (str): Name of the tool being exported
    '''

    all_frames = sorted(set(annotations.keys()) | set(predictions.keys()))
    rows = []

    for frame_idx in all_frames:
        time_sec = frame_idx / fps

        # Grab prediction
        pred_bbox = None
        if frame_idx in predictions and predictions[frame_idx]:
            bbox = predictions[frame_idx][0]['bbox']
            score = predictions[frame_idx][0]['score']
            pred_bbox = {'xmin': bbox[0], 'ymin': bbox[1], 'xmax': bbox[2], 'ymax': bbox[3], 'score': score}

        # Grab ground truth
        gt_bbox = None
        if frame_idx in annotations:
            for item in annotations[frame_idx]:
                if item['class'] == tool_class:
                    gt_bbox = item['bbox']
                    break

        # Grab IoU for frame
        iou = 0.0
        if gt_bbox and pred_bbox:
            iou = calculate_iou(gt_bbox, [pred_bbox['xmin'], pred_bbox['ymin'],
                                          pred_bbox['xmax'], pred_bbox['ymax']])

        rows.append({
            'frame': frame_idx,
            'time': f"{time_sec:.2f}",
            'tool_class': tool_class,
            'gt_bbox': str(gt_bbox) if gt_bbox else '[]',
            'pred_bbox': str([pred_bbox['xmin'], pred_bbox['ymin'],
                              pred_bbox['xmax'], pred_bbox['ymax']]) if pred_bbox else '[]',
            'pred_score': f"{pred_bbox['score']:.6f}" if pred_bbox else '0.0',
            'iou': f"{iou:.4f}"
        })

    # Write
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['frame', 'time', 'tool_class',
                                               'gt_bbox', 'pred_bbox', 'pred_score', 'iou'])
        writer.writeheader()
        writer.writerows(rows)


def export_metrics_csv(all_metrics, output_path):
    '''
    Writes performance metrics to CSV.

    Args:
        all_metrics (dict): Dictionary of metrics for all tools + overall
        output_path (Path): Destination file path
    '''

    rows = []
    for tool_class, metrics in all_metrics.items():
        rows.append({
            'tool': tool_class,
            'mAP@50': f"{metrics['mAP@50']:.4f}",
            'mAP@75': f"{metrics['mAP@75']:.4f}",
            'mAP@50-95': f"{metrics['mAP@50-95']:.4f}",
            'precision': f"{metrics['precision@50']:.4f}",
            'recall': f"{metrics['recall@50']:.4f}",
            'f1': f"{metrics['f1@50']:.4f}",
            'frames': metrics['num_frames']
        })

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['tool', 'mAP@50', 'mAP@75', 'mAP@50-95',
                                               'precision', 'recall', 'f1', 'frames'])
        writer.writeheader()
        writer.writerows(rows)


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


def process_datasets_yolo(participants=None, trials=None,
                          output_dir='./yolo26_results_fixed',
                          model_path='yolo26s.pt'):
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

    config = DatasetConfig()
    all_results = {}

    # LOOCV: each participant takes a turn as the held-out test set
    for test_participant in participants:
        print(f"LOOCV Fold: Test = {test_participant}")

        fold_dir = Path(output_dir) / f"Fold_{test_participant}"
        os.makedirs(fold_dir, exist_ok=True)

        # Everyone except the test participant trains
        train_participants = [p for p in participants if p != test_participant]
        print(f"Train/Val: {train_participants} (80/20 frame split); Test: {test_participant}")

        # Prepare YOLO dataset for this fold and train
        yaml_path = prepare_yolo_dataset(config, train_participants, trials, output_dir,
                                         CLASS_MAPPING, test_participant)
        tracker = YOLOTracker(model_path)
        tracker.train(str(yaml_path), epochs=100, project_dir=str(fold_dir))

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

            # Build tool info
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

            # Calculate metrics per tool
            all_predictions = {}
            all_metrics = {}

            for tool_class in sorted(all_tools):
                tool_predictions = all_tool_predictions.get(tool_class, {})

                # Add to combined predictions
                for frame_idx, preds in tool_predictions.items():
                    for pred in preds:
                        pred['tool_class'] = tool_class

                    if frame_idx not in all_predictions:
                        all_predictions[frame_idx] = []
                    all_predictions[frame_idx].extend(preds)

                # Calculate metrics
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

            # Overall metrics
            overall_metrics = calculate_metrics(annotations, all_predictions)
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


main()