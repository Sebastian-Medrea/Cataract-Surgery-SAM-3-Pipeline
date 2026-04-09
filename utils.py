"""
Shared utility functions for the Cataract Surgery Tool Tracking Pipeline.

Contains common operations used across SAM 3, YOLO, and analysis scripts:
config loading, dataset path construction, ground truth parsing, bounding box
conversion, IoU, mAP computation, motion metrics, and CSV export helpers.

CISC 499 
Sebastian Medrea 
Supervisor: Dr. Rebecca Hisey
"""

import ast
import csv
import numpy as np
import cv2
from pathlib import Path
import yaml


def load_config(config_path="config.yaml"):
    '''
    Load pipeline configuration from YAML file.

    Args:
        config_path (str): Path to config.yaml

    Returns:
        dict: Parsed configuration dictionary
    '''

    config_file = Path(config_path)

    if not config_file.exists():
        
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            f"Copy config.yaml and edit paths for your setup."
        )

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    return config


def get_tool_classes(config):
    '''
    Get sorted list of tool classes from config.
    '''
    return sorted(config['tool_classes'])


def get_class_mapping(config):
    '''
    Build integer ID to class name mapping for YOLO format.

    Returns:
        tuple: (class_mapping {int: str}, inv_mapping {str: int})
    '''

    tools = get_tool_classes(config)
    class_mapping = {i: name for i, name in enumerate(tools)}
    inv_mapping = {name: i for i, name in enumerate(tools)}

    return class_mapping, inv_mapping


class DatasetConfig:
    '''
    Handles path construction for the dataset structure.
    Accounts for P1's revised label naming convention.
    '''

    def __init__(self, config=None):

        if config is not None:
            base = config['dataset']['base_path']
            self.base_path = Path(base)
            self.video_dir = self.base_path / config['dataset']['video_subdir']
            self.dataset_dir = self.base_path / config['dataset']['label_subdir']

        else:
            self.base_path = Path("P:/data/PerkTutor/CataractSurgery")
            self.video_dir = self.base_path / "Videos/Simulated_Data/mkv"
            self.dataset_dir = self.base_path / "Datasets/Simulated_Data"

    def get_video_path(self, participant, trial):
        '''
        Construct path to video file.

        Args:
            participant (str): Participant ID (e.g. 'P1')
            trial (int): Trial number (e.g. 1)

        Returns:
            Path: Full path to .mkv video file
        '''
        return self.video_dir / f"{participant}.{trial}-1.mkv"

    def get_csv_path(self, participant, trial):
        '''
        Construct path to ground truth CSV labels.
        P1 uses revised labels with different folder/file naming.
        '''

        if participant == 'P1':
            folder = self.dataset_dir / f"P1_{trial}_Revised"
            csv_name = f"P1_{trial}_Revised_Labels.csv"

        else:
            folder = self.dataset_dir / f"{participant}_{trial}"
            csv_name = f"{participant}_{trial}_Labels.csv"

        return folder / csv_name


def load_annotations(csv_path, fps):
    '''
    Parse ground truth bounding box CSV into frame-indexed dictionary.
    Filters out eye and lens classes since we only track instruments.

    Args:
        csv_path (Path): Path to ground truth .csv file
        fps (float): Video frame rate for time-to-frame conversion

    Returns:
        dict: {frame_index: [{'bbox': [x1,y1,x2,y2], 'class': str}, ...]}
    '''

    annotations = {}

    try:
        with open(csv_path, 'r') as f:
            for row in csv.DictReader(f):

                # Convert timestamp to frame index and parse the bbox list string
                frame_idx = int(round(float(row['Time Recorded']) * fps))
                bbox_str = row['Tool bounding box']

                if bbox_str == '[]':
                    continue

                bboxes = ast.literal_eval(bbox_str)
                frame_tools = []

                for bbox in bboxes:
                    class_name = bbox['class'].lower().replace('_', ' ').strip()

                    # Skip non-instrument classes
                    if class_name in ('eye', 'lens'):
                        continue

                    frame_tools.append({
                        'bbox': [bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']],
                        'class': class_name
                    })

                if frame_tools:
                    annotations[frame_idx] = frame_tools

    except FileNotFoundError:
        print(f"  Warning: Annotation file not found: {csv_path}")

    except Exception as e:
        print(f"  Error loading annotations from {csv_path}: {e}")

    return annotations


def get_video_fps(video_path):
    '''
    Read FPS from video file metadata.
    '''

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    return fps



def parse_bbox(bbox_str):
    '''
    Safely parse a bounding box string from CSV back to a list.
    Handles empty, nan, None, or malformed strings.

    Args:
        bbox_str (str): String like '[10, 20, 100, 200]'

    Returns:
        list or None: [xmin, ymin, xmax, ymax] if valid, None otherwise
    '''

    s = str(bbox_str).strip()

    if s in ('[]', '', 'nan', 'None'):
        return None

    try:

        bbox = ast.literal_eval(s)
        if isinstance(bbox, list) and len(bbox) == 4:
            return bbox

    except (ValueError, SyntaxError):
        pass

    return None


def xyxy_to_yolo(img_size, bbox):
    '''
    Convert [xmin, ymin, xmax, ymax] pixel coordinates to YOLO normalized format
    [x_center, y_center, width, height] where all values are 0-1.

    Args:
        img_size (tuple): Image shape (height, width, channels)
        bbox (list): [xmin, ymin, xmax, ymax]

    Returns:
        tuple: (x_center, y_center, width, height) normalized
    '''

    dw = 1.0 / img_size[1]
    dh = 1.0 / img_size[0]

    x_center = (bbox[0] + bbox[2]) / 2.0 * dw
    y_center = (bbox[1] + bbox[3]) / 2.0 * dh
    w = (bbox[2] - bbox[0]) * dw
    h = (bbox[3] - bbox[1]) * dh

    return x_center, y_center, w, h


def calculate_bbox_center(bbox):
    '''
    Compute the 2D center of a bounding box.
    Same format as calculate_video_skill_metrics_no_depth.py.
    '''
    x_center = bbox[0] + (bbox[2] - bbox[0]) / 2.0
    y_center = bbox[1] + (bbox[3] - bbox[1]) / 2.0
    return (x_center, y_center)


def euclidean_distance(point1, point2):
    '''
    Euclidean distance between two 2D points.
    '''
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)


def calculate_iou(bbox1, bbox2):
    '''
    Intersection over Union between two bounding boxes.

    Args:
        bbox1 (list): [xmin, ymin, xmax, ymax]
        bbox2 (list): [xmin, ymin, xmax, ymax]

    Returns:
        float: IoU score (0.0 to 1.0)
    '''

    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)

    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = bbox1_area + bbox2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def calculate_metrics(ground_truth, predictions, eval_mode="all_gt"):
    '''
    Compute mAP@50, mAP@75, mAP@50-95, precision, recall, and F1.

    Two evaluation modes:
    - "all_gt": evaluate on ALL ground truth frames (YOLO, missing preds = misses)
    - "common": evaluate only on frames where both GT and predictions exist (SAM 3)

    Args:
        ground_truth (dict): {frame_idx: [{'bbox': [...], ...}, ...]}
        predictions (dict): {frame_idx: [{'bbox': [...], 'score': float, ...}, ...]}
        eval_mode (str): "all_gt" or "common"

    Returns:
        dict: Detection metrics
    '''

    # thresholds from 0.50 to 0.95 in steps of 0.05
    iou_thresholds = np.linspace(0.5, 0.95, 10)

    # "common" only evaluates frames where both GT and predictions exist 
    # "all_gt" penalizes missing predictions
    if eval_mode == "common":
        eval_frames = sorted(set(ground_truth.keys()) & set(predictions.keys()))

    else:
        eval_frames = sorted(ground_truth.keys())

    if not eval_frames:
        return {
            'mAP@50': 0.0, 'mAP@75': 0.0, 'mAP@50-95': 0.0,
            'precision@50': 0.0, 'recall@50': 0.0, 'f1@50': 0.0,
            'num_frames': 0
        }

    aps_per_threshold = {}

    # Compute Average Precision at each IoU threshold
    for iou_thresh in iou_thresholds:

        # Collect all predictions and GT counts across all eval frames
        all_preds = []
        total_gt_boxes = 0

        for frame_idx in eval_frames:
            gt_bboxes = [item['bbox'] for item in ground_truth.get(frame_idx, [])]
            total_gt_boxes += len(gt_bboxes)

            for pred in predictions.get(frame_idx, []):
                all_preds.append({
                    'frame': frame_idx, 'bbox': pred['bbox'], 'score': pred['score']
                })

        # Sort by confidence (highest first) for precision-recall curve
        all_preds = sorted(all_preds, key=lambda x: x['score'], reverse=True)
        true_positives = np.zeros(len(all_preds))
        false_positives = np.zeros(len(all_preds))
        matched_gts = {f: set() for f in eval_frames}

        # Greedy matching: each prediction tries to match the best unmatched GT box
        for pred_idx, pred in enumerate(all_preds):

            frame_idx = pred['frame']
            gt_bboxes = [item['bbox'] for item in ground_truth.get(frame_idx, [])]
            max_iou = 0
            max_gt_idx = -1

            for gt_idx, gt_bbox in enumerate(gt_bboxes):

                if gt_idx in matched_gts[frame_idx]:
                    continue

                iou = calculate_iou(pred['bbox'], gt_bbox)

                if iou > max_iou:
                    max_iou = iou
                    max_gt_idx = gt_idx

            if max_iou >= iou_thresh:
                true_positives[pred_idx] = 1
                matched_gts[frame_idx].add(max_gt_idx)

            else:
                false_positives[pred_idx] = 1

        # Compute AP from the precision-recall curve 
        if total_gt_boxes == 0:
            ap = 0.0

        else:

            tp_cumsum = np.cumsum(true_positives)
            fp_cumsum = np.cumsum(false_positives)
            recalls = tp_cumsum / total_gt_boxes
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum + np.finfo(float).eps)

            # Pad with values for the boundaries of the PR curve
            precisions = np.concatenate(([0.0], precisions, [0.0]))
            recalls = np.concatenate(([0.0], recalls, [1.0]))

            # Decreasing precision (right-to-left max) for interpolation
            for i in range(len(precisions) - 1, 0, -1):
                precisions[i - 1] = max(precisions[i - 1], precisions[i])

            # Sum rectangular areas under the interpolated PR curve
            indices = np.where(recalls[1:] != recalls[:-1])[0] + 1
            ap = sum((recalls[i] - recalls[i - 1]) * precisions[i] for i in indices)

        aps_per_threshold[round(iou_thresh, 2)] = ap

    # Separate pass for Precision / Recall / F1 at IoU = 0.5 
    total_tp = total_fp = total_fn = 0

    for frame_idx in eval_frames:
        gt_bboxes = [item['bbox'] for item in ground_truth.get(frame_idx, [])]
        pred_items = predictions.get(frame_idx, [])

        # No predictions on this frame means all GT boxes are missed
        if not pred_items:
            total_fn += len(gt_bboxes)
            continue

        # Match each prediction to its best GT box at IoU >= 0.5
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

        # Any unmatched GT boxes are false negatives
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


def compute_motion_for_tool(pred_csv_path, gap_threshold=5.0, fps=15.0):
    '''
    Compute path length (px) and usage time (s) from a per-tool prediction CSV.
    Skips detection gaps exceeding gap_threshold to avoid inflation.

    Args:
        pred_csv_path (Path): Path to prediction CSV
        gap_threshold (float): Max seconds between consecutive detections before skipping
        fps (float): Video frame rate for usage time calculation

    Returns:
        dict: {'path_length': float, 'usage_time': float, 'num_detections': int}
    '''

    empty = {'path_length': 0.0, 'usage_time': 0.0, 'num_detections': 0}

    if not Path(pred_csv_path).exists():
        return empty

    try:

        import pandas as pd
        df = pd.read_csv(pred_csv_path)

    except Exception:
        return empty

    # Extract bbox centers from each row that has a valid predicted bbox
    detections = []

    for _, row in df.iterrows():
        bbox = parse_bbox(row['pred_bbox'])

        if bbox is not None:
            center = calculate_bbox_center(bbox)
            detections.append({'time': float(row['time']), 'center': center})

    if len(detections) == 0:
        return empty

    # Accumulate Euclidean path length between consecutive detections, but skip gaps longer than the threshold to avoid inflating the path
    # (e.g. tool leaves frame and reappears elsewhere)
    path_length = 0.0

    for i in range(1, len(detections)):
        time_delta = detections[i]['time'] - detections[i - 1]['time']

        if 0 < time_delta < gap_threshold:
            path_length += euclidean_distance(detections[i - 1]['center'], detections[i]['center'])

    return {
        'path_length': path_length,
        'usage_time': len(detections) / fps,
        'num_detections': len(detections)
    }


def export_predictions_csv(predictions, annotations, fps, output_path, tool_class):
    '''
    Write frame-by-frame tracking results to CSV with GT comparison.

    Args:
        predictions (dict): Model predictions per frame
        annotations (dict): Ground truth annotations per frame
        fps (float): Video FPS
        output_path (Path): Destination CSV path
        tool_class (str): Name of the tool being exported
    '''

    # Union of GT and predicted frames so we capture misses and false positives
    all_frames = sorted(set(annotations.keys()) | set(predictions.keys()))
    rows = []

    for frame_idx in all_frames:
        time_sec = frame_idx / fps

        # Extract best prediction for this frame (take first if multiple)
        pred_bbox = None

        if frame_idx in predictions and predictions[frame_idx]:
            bbox = predictions[frame_idx][0]['bbox']
            score = predictions[frame_idx][0]['score']
            pred_bbox = {
                'xmin': bbox[0], 'ymin': bbox[1],
                'xmax': bbox[2], 'ymax': bbox[3], 'score': score
            }

        # Find matching ground truth for this specific tool
        gt_bbox = None

        if frame_idx in annotations:
            for item in annotations[frame_idx]:

                if item['class'] == tool_class:

                    gt_bbox = item['bbox']
                    break

        # Compute IoU between GT and prediction for this frame
        iou = 0.0

        if gt_bbox and pred_bbox:
            iou = calculate_iou(
                gt_bbox,
                [pred_bbox['xmin'], pred_bbox['ymin'], pred_bbox['xmax'], pred_bbox['ymax']]
            )

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

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'frame', 'time', 'tool_class', 'gt_bbox', 'pred_bbox', 'pred_score', 'iou'
        ])
        writer.writeheader()
        writer.writerows(rows)


def export_metrics_csv(all_metrics, output_path):
    '''
    Write per-tool detection metrics to CSV.

    Args:
        all_metrics (dict): {tool_name: metrics_dict}
        output_path (Path): Destination CSV path
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
        writer = csv.DictWriter(f, fieldnames=[
            'tool', 'mAP@50', 'mAP@75', 'mAP@50-95', 'precision', 'recall', 'f1', 'frames'
        ])
        writer.writeheader()
        writer.writerows(rows)


def build_tool_info(annotations, masks_dir=None):
    '''
    Build tool info dictionary from annotations: first/last frame, init bbox.
    Used by both SAM 3 (needs init_bbox for prompting) and YOLO (needs frame windows).

    Args:
        annotations (dict): Loaded ground truth annotations
        masks_dir (Path or None): Base directory for mask/frame storage

    Returns:
        dict: {tool_class: {'first_frame', 'last_frame', 'init_bbox', 'mask_dir'}}
    '''

    all_tools = set()

    for frame_data in annotations.values():
        for item in frame_data:

            all_tools.add(item['class'])

    tool_info = {}

    for tool_class in sorted(all_tools):
        # Find all frames where this tool appears in the ground truth
        tool_frames = sorted([
            f for f, items in annotations.items()
            if any(item['class'] == tool_class for item in items)
        ])

        if not tool_frames:
            continue

        # Grab the GT bbox from the first appearance as the initial prompt for SAM 3
        first_frame = tool_frames[0]
        init_bbox = next(
            item['bbox'] for item in annotations[first_frame]
            if item['class'] == tool_class
        )

        info = {
            'first_frame': tool_frames[0],
            'last_frame': tool_frames[-1],
            'init_bbox': init_bbox,
        }

        if masks_dir is not None:
            info['mask_dir'] = masks_dir / tool_class.replace(' ', '_')

        tool_info[tool_class] = info

    return tool_info