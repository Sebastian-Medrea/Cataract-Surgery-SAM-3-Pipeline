"""
SAM 3 Zero-Shot Inference on Surgical Tool Object Detection With Bounding Box Prompting for Cataract Surgery.

CISC 499
Student: Sebastian Medrea
Supervisor: Dr. Rebecca Hisey

This program uses a zero-shot tracking pipeline from the new Segment Anything Model 3 (SAM 3) to test the feasibility 
of SAM 3 for surgical tool tracking. Removing the annotation burden required and/or increase access to skill assessment
metrics through cheaper deep learning alternatives with an automated inference to save time and resources and be used for further implementation such as surgical
skill assessment of novices/medical school trainees. 

mAP50 metrics, IoU, amongst others are compared against ground truth bounding boxes and the predicted SAM 3 bounding boxes.

Inputs:
- Videos: .mkv files from the PerkTutor CataractSurgery dataset
- Labels: .csv files with ground truth bounding boxes

Outputs:
- predictions.csv: Frame-by-frame tracking results
- metrics.csv: Evaluations (mAP, Precision, Recall)
- visualizations/: PNG overlay of masks
"""

import ast
import csv
import gc
import glob
from pathlib import Path
import traceback
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import torch
import time

# HuggingFace Transformers for SAM 3
from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
from accelerate import Accelerator

class DatasetConfig:
    '''
    Handles path differences between P1-P5 datasets for the Perk Lab dataset structure with Revised and
    normal labels. (Swapped through some desktops so there was not much time to download the extensive files
    so I pulled from the network drive for training; will download at a later date.)
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

class SAM3Tracker:
    '''
    Wrapper for SAM 3 video tracking model from HuggingFace.

    Model weights are loaded onto GPU, video frames are preprocessed, and the inference session memory banks are set
    for tracking.
    '''
    
    def __init__(self):

        # Use NVIDIA GPU
        self.device = Accelerator().device if torch.cuda.is_available() else "cpu"
        print(f"Device: {self.device}")
        
        # Load model in bfloat16 to save VRAM
        self.model = Sam3TrackerVideoModel.from_pretrained("facebook/sam3").to(self.device, dtype=torch.bfloat16)
        
        self.processor = Sam3TrackerVideoProcessor.from_pretrained("facebook/sam3")
        print("SAM 3 loaded.")
    
    def _load_frames(self, video_path, start_frame, num_frames):
        '''
        Reads a specific chunk of frames from video file to avoid running out of RAM.
        '''

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return []
        
        # Seek to the start of the chunk
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frames = []
        for i in range(num_frames):
            ret, frame = cap.read()

            # End of video
            if not ret:
                break

            # OpenCV in BGR, convert to RGB for model
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        cap.release()
        return frames
    
    def track_sequential(self, video_path, tool_info_dict, chunk_size=100):
        '''
        Track one tool at a time through video, then reset for the next tool for mem management.

        Args:
            video_path (Path): Path to the video
            tool_info_dict (dict): Dictionary containing start frames and init boxes for each tool
            chunk_size (int): Number of frames to process in VRAM at once
            
        Returns:
            dict: Nested dictionary of predictions {tool_name: {frame_idx: [predictions]}}.
        '''

        all_predictions = {}
        
        # Sort tools alphabetically for consistency
        for tool_idx, (tool_class, info) in enumerate(sorted(tool_info_dict.items())):
            print(f"[{tool_idx+1}/{len(tool_info_dict)}] {tool_class}")
            
            # Process specific tool completely before moving to the next
            tool_predictions = self._track_single_tool(
                video_path=video_path,
                first_frame=info['first_frame'],
                last_frame=info['last_frame'],
                init_bbox=info['init_bbox'],
                mask_dir=info['mask_dir'],
                chunk_size=chunk_size
            )
            
            all_predictions[tool_class] = tool_predictions

            # Debug for tool detection
            num_detections = sum(1 for p in tool_predictions.values() if p)
            print(f"{num_detections} detections")
        
        return all_predictions
    
    
    def track_simultaneous(self, video_path, tool_info_dict, chunk_size=100):
        '''
        Track all tools in a single pass through the video instead of one at a time.

        Args:
            video_path (Path): Path to the video
            tool_info_dict (dict): Dictionary containing start frames and init boxes for each tool
            chunk_size (int): Number of frames to process in VRAM at once
            
        Returns:
            dict: Nested dictionary of predictions {tool_name: {frame_idx: [predictions]}}.
        '''
        '''
        print(f"Tracking {len(tool_info_dict)} tools simultaneously")
        
        # Find global start and end frames for video
        all_first = [info['first_frame'] for info in tool_info_dict.values()]
        all_last = [info['last_frame'] for info in tool_info_dict.values()]
        global_start = min(all_first)
        global_end = max(all_last)

        # Break in chunks
        total_frames = global_end - global_start + 1
        num_chunks = (total_frames + chunk_size - 1) // chunk_size
        
        print(f"Frames {global_start}-{global_end} in {num_chunks} chunks")
        
        all_predictions = {tool: {} for tool in tool_info_dict.keys()}
        
        # Map tool names (str) to integer IDs (1,2, ..) since model needs object and query to intialize inference with multiple bboxes
        tool_list = sorted(tool_info_dict.keys())
        tool_to_obj_id = {tool: i+1 for i, tool in enumerate(tool_list)}

        for chunk_idx in range(num_chunks):

            # Calculate start/end frames for specific chunk
            chunk_start = global_start + (chunk_idx * chunk_size)
            chunk_frames_count = min(chunk_size, global_end - chunk_start + 1)
            
            # Load specific frames
            frames = self._load_frames(video_path, chunk_start, chunk_frames_count)
            if not frames:
                continue
            
            # Initialize new inference session (resets memory bank; may interfere with the carry over tools)
            session = self.processor.init_video_session(
                video=frames,
                inference_device=self.device,
                dtype=torch.bfloat16
            )
            
            # Determine which tools need prompts in this chunk
            prompts_by_frame = {} # frame_idx to list of (obj_id, bbox)
            
            # For each tool, check if it exists in the current chunk time and init if present or reset session if tool continues from prev chunk
            for tool_class, info in tool_info_dict.items():
                obj_id = tool_to_obj_id[tool_class]
                
                # Check if tool is active in this chunk
                if info['first_frame'] <= chunk_start + chunk_frames_count - 1 and info['last_frame'] >= chunk_start:
                    
                    # Determine initialization logic
                    init_bbox = None
                    relative_frame_idx = -1
                    
                    # Tool starts naturally inside this chunk
                    if info['first_frame'] >= chunk_start:
                        relative_frame_idx = info['first_frame'] - chunk_start
                        init_bbox = info['init_bbox']
                        
                    # Tool is carried over from previous chunk
                    # Re-prompt at frame 0 of this chunk using the last prediction
                    else:
                        relative_frame_idx = 0

                        # Try to get the last prediction from previous processing
                        if all_predictions[tool_class]:
                            last_known_frame = max(all_predictions[tool_class].keys())

                            if all_predictions[tool_class][last_known_frame]:
                                init_bbox = all_predictions[tool_class][last_known_frame][0]['bbox']

                            else:

                                # Fallback to init if tracking lost
                                init_bbox = info['init_bbox'] 
                        else:
                            init_bbox = info['init_bbox']

                    # Add prompt to queue
                    if init_bbox is not None:
                        if relative_frame_idx not in prompts_by_frame:

                            prompts_by_frame[relative_frame_idx] = []
                        prompts_by_frame[relative_frame_idx].append((obj_id, init_bbox))

            # SAM 3 needs to run inference on the frame where a prompt is added to generate memory features or it will crash
            sorted_prompt_frames = sorted(prompts_by_frame.keys())
            
            # No active tools
            if not sorted_prompt_frames:
                continue

            for frame_idx in sorted_prompt_frames:

                # Input to processor
                frame_prompts = prompts_by_frame[frame_idx]
                p_obj_ids = [p[0] for p in frame_prompts]
                p_boxes = [[p[1]] for p in frame_prompts] # Shape needs to be [num_objects, num_prompts, 4]
                
                # Add boxes and for the session and run memory encoder right after for the new prompts
                self.processor.add_inputs_to_inference_session(
                    inference_session=session,
                    frame_idx=frame_idx,
                    obj_ids=p_obj_ids,
                    input_boxes=p_boxes
                )
                
                _ = self.model(inference_session=session, frame_idx=frame_idx)
            
            # Propogate from earliest frame with an intialized memory bank of prompts, convert model tensor to binary mask, and map back to tool names by index
            for sam3_output in self.model.propagate_in_video_iterator(session):
                frame_idx = sam3_output.frame_idx + chunk_start
                
                masks = self.processor.post_process_masks(
                    [sam3_output.pred_masks],
                    original_sizes=[[session.video_height, session.video_width]],
                    binarize=True
                )[0]
                
                # Extract bbox for each tool
                # session.obj_ids maps the model's internal index to provided obj_id
                current_session_obj_ids = list(session.obj_ids)
                
                for tool_class, obj_id in tool_to_obj_id.items():

                    if obj_id in current_session_obj_ids:
                        obj_idx = current_session_obj_ids.index(obj_id)
                        
                        # Extract mask and convert to bounding box
                        if obj_idx < masks.shape[0]:
                            mask = masks[obj_idx, 0].cpu().numpy().astype(bool)
                            
                            rows = np.any(mask, axis=1)
                            cols = np.any(mask, axis=0)
                            
                            if rows.any() and cols.any():
                                ymin, ymax = np.where(rows)[0][[0, -1]]
                                xmin, xmax = np.where(cols)[0][[0, -1]]
                                score = float(mask.sum()) / mask.size
                                
                                all_predictions[tool_class][frame_idx] = [{
                                    'bbox': [int(xmin), int(ymin), int(xmax), int(ymax)],
                                    'score': score,
                                    'obj_id': obj_id
                                }]

            # Cleanup
            torch.cuda.empty_cache()
            gc.collect()
            
        return all_predictions
        '''


        print(f"Tracking {len(tool_info_dict)} tools simultaneously")
        
        # Find global start and end frames across all tools
        all_first = [info['first_frame'] for info in tool_info_dict.values()]
        all_last = [info['last_frame'] for info in tool_info_dict.values()]
        global_start = min(all_first)
        global_end = max(all_last)

        # Break into chunks for memory
        total_frames = global_end - global_start + 1
        num_chunks = (total_frames + chunk_size - 1) // chunk_size
        
        print(f"Frames {global_start}-{global_end} in {num_chunks} chunks")
        
        all_predictions = {tool: {} for tool in tool_info_dict.keys()}
        
        # Map tool names to integer IDs since the model needs numeric object IDs for multi-object tracking
        tool_list = sorted(tool_info_dict.keys())
        tool_to_obj_id = {tool: i + 1 for i, tool in enumerate(tool_list)}

        for chunk_idx in range(num_chunks):

            # Calculate start/end frames for this chunk
            chunk_start = global_start + (chunk_idx * chunk_size)
            chunk_frames_count = min(chunk_size, global_end - chunk_start + 1)
            
            # Load specific frames into memory
            frames = self._load_frames(video_path, chunk_start, chunk_frames_count)
            if not frames:
                continue
            
            # Initialize new inference session (resets memory bank each chunk)
            session = self.processor.init_video_session(
                video=frames,
                inference_device=self.device,
                dtype=torch.bfloat16
            )
            
            # Gather all prompts needed in this chunk, grouped by their relative frame index
            prompts_by_frame = {}
            active_tool_count = 0
            
            for tool_class, info in tool_info_dict.items():
                tool_active = (
                    info['first_frame'] <= chunk_start + chunk_frames_count - 1
                    and info['last_frame'] >= chunk_start
                )
                if not tool_active:
                    continue
                
                active_tool_count += 1
                
                # If tool appears naturally in this chunk, prompt at its specific frame
                if info['first_frame'] >= chunk_start:
                    rel_frame = info['first_frame'] - chunk_start
                    init_bbox = info['init_bbox']

                else:
                    # Tool carries over from previous chunk; prompt at frame 0 of this chunk
                    rel_frame = 0
                    preds = all_predictions[tool_class]
                    if preds:
                        last_known_frame = max(preds.keys())

                        if preds[last_known_frame]:
                            init_bbox = preds[last_known_frame][0]['bbox']
                        else:
                            init_bbox = info['init_bbox']

                    else:
                        init_bbox = info['init_bbox']
                        
                if rel_frame not in prompts_by_frame:
                    prompts_by_frame[rel_frame] = []
                prompts_by_frame[rel_frame].append((tool_class, init_bbox))

            if active_tool_count == 0:
                continue

            # Register all prompts at their correct frames
            sorted_prompt_frames = sorted(prompts_by_frame.keys())
            
            for rel_frame in sorted_prompt_frames:
                frame_obj_ids = []
                frame_boxes = []
                
                for tool_class, bbox in prompts_by_frame[rel_frame]:
                    frame_obj_ids.append(tool_to_obj_id[tool_class])
                    # Append the raw list [xmin, ymin, xmax, ymax]
                    frame_boxes.append(bbox) 
                
                # HuggingFace API expects input_boxes as [batch_size, num_objects, 4] so wrap in outer list
                batched_boxes = [frame_boxes] 
                
                self.processor.add_inputs_to_inference_session(
                    inference_session=session,
                    frame_idx=rel_frame,
                    obj_ids=frame_obj_ids,
                    input_boxes=batched_boxes
                )
                
                # Encode prompts into the memory bank for this frame
                _ = self.model(inference_session=session, frame_idx=rel_frame)
                
            # Propagate from the earliest prompted frame
            for sam3_output in self.model.propagate_in_video_iterator(session):
                frame_idx = sam3_output.frame_idx + chunk_start
                
                # Convert model tensor output to binary mask
                masks = self.processor.post_process_masks(
                    [sam3_output.pred_masks],
                    original_sizes=[[session.video_height, session.video_width]],
                    binarize=True
                )[0]
                
                # session.obj_ids maps the model's internal index to our provided obj_id
                current_session_obj_ids = list(session.obj_ids)
                
                for tool_class, obj_id in tool_to_obj_id.items():
                    if obj_id not in current_session_obj_ids:
                        continue
                        
                    # Skip frames outside the tool's active window
                    tool_first = tool_info_dict[tool_class]['first_frame']
                    tool_last = tool_info_dict[tool_class]['last_frame']

                    if frame_idx < tool_first or frame_idx > tool_last:
                        continue
                        
                    obj_idx = current_session_obj_ids.index(obj_id)
                    if obj_idx >= masks.shape[0]:
                        continue
                    
                    # Extract mask and convert to bounding box
                    mask = masks[obj_idx, 0].cpu().numpy().astype(bool)
                    
                    rows = np.any(mask, axis=1)
                    cols = np.any(mask, axis=0)
                    
                    if rows.any() and cols.any():
                        ymin, ymax = np.where(rows)[0][[0, -1]]
                        xmin, xmax = np.where(cols)[0][[0, -1]]
                        score = float(mask.sum()) / mask.size
                        
                        all_predictions[tool_class][frame_idx] = [{
                            'bbox': [int(xmin), int(ymin), int(xmax), int(ymax)],
                            'score': score,
                            'obj_id': obj_id
                        }]

                        mask_dir = tool_info_dict[tool_class]['mask_dir']
                        os.makedirs(mask_dir, exist_ok=True)
                        
                        # Save mask
                        np.save(f'{mask_dir}/mask_{frame_idx:04d}.npy', mask)
                        
                        # Save the original frame (using the relative chunk index)
                        np.save(f'{mask_dir}/frame_{frame_idx:04d}.npy', frames[sam3_output.frame_idx])

            # Cleanup VRAM between chunks
            del session
            del frames
            if 'masks' in locals():
                del masks

            torch.cuda.empty_cache()
            gc.collect()
            
        return all_predictions
    
def load_annotations(csv_path, fps):
    '''
    Grabs ground truth CSV files for bounding boxes from dataset

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

            # CSV is logged in time, so multiply by FPS (15) to get frame
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
                if bbox['class'] != 'eye':
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

def calculate_iou(bbox1, bbox2):
    '''
    Calculate intersection over union between ground truth and predicted bounding box (area of overlap/area of union)

    Args:
        bbox1 (list): [xmin, ymin, xmax, ymax]
        bbox2 (list): [xmin, ymin, xmax, ymax]
        
    Returns:
        float: The IoU score (0.0 to 1.0)
    '''

    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    #Coordinates of intersection rectangle
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
    Computes object detection metrics for tools.
    Calculates mAP (Mean Average Precision) at various IoU thresholds (0.5 to 0.95).
    
    Args:
        ground_truth (dict): Dictionary of ground truth annotations
        predictions (dict): Dictionary of model predictions
        
    Returns:
        dict: Dictionary containing 'mAP@50', 'mAP@75', 'f1', etc.
    '''
    iou_thresholds = np.linspace(0.5, 0.95, 10)

    # Calculate metrics for frames only with data
    common_frames = sorted(set(ground_truth.keys()) & set(predictions.keys()))
    
    if not common_frames:
        return {
            'mAP@50': 0.0, 'mAP@75': 0.0, 'mAP@50-95': 0.0,
            'precision@50': 0.0, 'recall@50': 0.0, 'f1@50': 0.0,
            'num_frames': 0
        }
    
    aps_per_threshold = {}
    
    for iou_thresh in iou_thresholds:
        frame_aps = []
        for frame_idx in common_frames:
            gt_bboxes = [item['bbox'] for item in ground_truth[frame_idx]]
            pred_items = predictions[frame_idx]
            
            if not gt_bboxes or not pred_items:
                frame_aps.append(0.0)
                continue
            
            # Sort predictions by confidence score for highest first
            pred_items = sorted(pred_items, key=lambda x: x['score'], reverse=True)
            true_positives = []
            matched_gt = set()
            
            for pred in pred_items:
                max_iou = 0
                max_gt_idx = -1

                # Check prediction against all ground truth boxes
                for gt_idx, gt_bbox in enumerate(gt_bboxes):

                    # Matched
                    if gt_idx in matched_gt:
                        continue

                    iou = calculate_iou(pred['bbox'], gt_bbox)
                    if iou > max_iou:
                        max_iou = iou
                        max_gt_idx = gt_idx
                
                # Target hit
                if max_iou >= iou_thresh:
                    true_positives.append(1)
                    matched_gt.add(max_gt_idx)

                else:
                    true_positives.append(0)
            
            # Calculate Average Precision for frame
            if true_positives:

                tp_cumsum = np.cumsum(true_positives)
                precision = tp_cumsum / np.arange(1, len(true_positives) + 1)
                recall = tp_cumsum / len(gt_bboxes)

                ap = sum(precision[i] * (recall[i] - (recall[i-1] if i > 0 else 0)) 
                        for i in range(len(recall)))
                frame_aps.append(ap)

            else:
                frame_aps.append(0.0)
        
        aps_per_threshold[round(iou_thresh, 2)] = np.mean(frame_aps)
    
    # Precision/Recall/F1 at IoU=0.5 
    total_tp = total_fp = total_fn = 0

    for frame_idx in common_frames:

        gt_bboxes = [item['bbox'] for item in ground_truth[frame_idx]]
        pred_items = predictions[frame_idx]
        
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
        'num_frames': len(common_frames)
    }

def export_predictions_csv(predictions, annotations, fps, output_path, tool_class):
    '''
    Writes the frame-by-frame tracking results to CSV
    
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
        writer = csv.DictWriter(f, fieldnames=['frame', 'time', 'tool_class', 'gt_bbox', 
                                               'pred_bbox', 'pred_score', 'iou'])
        writer.writeheader()
        writer.writerows(rows)

def export_metrics_csv(all_metrics, output_path):
    '''
    Writes performance metrics to CSV

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

def visualize_tool_masks(tool_name, annotations, predictions, mask_dir, output_dir):
    '''
    Generates PNG images comparing Ground Truth vs. SAM 3 Prediction
    
    Args:
        tool_name (str): Name of the tool ('Forceps')
        annotations (dict): Ground truth data
        predictions (dict): Model predictions
        mask_dir (Path): Source directory of saved numpy masks
        output_dir (Path): Destination directory for PNGs
    '''
    
    '''
    # Find all npy masks
    mask_files = sorted(glob.glob(f'{mask_dir}/mask_*.npy'))
    if not mask_files:
        return
    
    print(f"Visualizing {len(mask_files)} masks for {tool_name}.")
    
    for mask_file in mask_files:

        # Grab frame number from file name
        frame_num = int(Path(mask_file).stem.split('_')[-1])
        
        # Load original image frame and predicted mask
        frame = np.load(f'{mask_dir}/frame_{frame_num:04d}.npy')
        mask = np.load(mask_file)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Ground truth
        axes[0].imshow(frame)
        if frame_num in annotations:
            for item in annotations[frame_num]:
                bbox = item['bbox']
                axes[0].add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                                  fill=False, edgecolor='lime', linewidth=2))
        axes[0].set_title(f'Frame {frame_num} - Ground Truth')
        axes[0].axis('off')
        
        # Mask
        axes[1].imshow(mask, cmap='jet')
        axes[1].set_title(f'SAM 3 Mask - {tool_name}')
        axes[1].axis('off')
        
        # Prediction
        axes[2].imshow(frame)
        if frame_num in predictions and predictions[frame_num]:
            bbox = predictions[frame_num][0]['bbox']
            axes[2].add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                              fill=False, edgecolor='cyan', linewidth=2))
            
            if frame_num in annotations:
                for gt_item in annotations[frame_num]:
                    if gt_item['class'] == tool_name:
                        gt = gt_item['bbox']
                        iou = calculate_iou(gt, bbox)
                        axes[2].text(10, 30, f'IoU: {iou:.3f}', color='white', fontsize=12,
                                   bbox=dict(facecolor='green' if iou > 0.5 else 'red', alpha=0.8))
                        break
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{tool_name}_frame{frame_num:04d}.png', dpi=100)
        plt.close()
        '''
    
    # Find all npy masks
    mask_files = sorted(glob.glob(f'{mask_dir}/mask_*.npy'))
    if not mask_files:
        return
    
    print(f"Visualizing {len(mask_files)} masks for {tool_name}.")
    
    for mask_file in mask_files:

        # Grab frame number from file name
        frame_num = int(Path(mask_file).stem.split('_')[-1])
        
        # Load original image frame and predicted mask
        frame = np.load(f'{mask_dir}/frame_{frame_num:04d}.npy')
        mask = np.load(mask_file)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Ground truth
        axes[0].imshow(frame)
        if frame_num in annotations:
            for item in annotations[frame_num]:
                bbox = item['bbox']
                axes[0].add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                                  fill=False, edgecolor='lime', linewidth=2))
                
                # Draw class label above bbox
                axes[0].text(bbox[0], max(0, bbox[1] - 8), item['class'], color='lime', 
                             fontsize=10, fontweight='bold', bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=2))
                
        axes[0].set_title(f'Frame {frame_num} - Ground Truth')
        axes[0].axis('off')
        
        # Mask
        axes[1].imshow(mask, cmap='jet')
        axes[1].set_title(f'SAM 3 Mask - {tool_name}')
        axes[1].axis('off')
        
        # Prediction
        axes[2].imshow(frame)
        if frame_num in predictions and predictions[frame_num]:
            bbox = predictions[frame_num][0]['bbox']
            axes[2].add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                              fill=False, edgecolor='cyan', linewidth=2))
            
            # Draw predicted class label above bbox
            axes[2].text(bbox[0], max(0, bbox[1] - 8), tool_name, color='cyan', 
                         fontsize=10, fontweight='bold', bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=2))
            
            if frame_num in annotations:
                for gt_item in annotations[frame_num]:
                    if gt_item['class'] == tool_name:
                        gt = gt_item['bbox']
                        iou = calculate_iou(gt, bbox)
                        # IoU text at bottom left to avoid overlapping tool label
                        axes[2].text(10, frame.shape[0] - 30, f'IoU: {iou:.3f}', color='white', fontsize=12,
                                   bbox=dict(facecolor='green' if iou > 0.5 else 'red', alpha=0.8))
                        break
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{tool_name}_frame{frame_num:04d}.png', dpi=100)
        plt.close()

def track_video(participant, trial, mode='sequential', config=None, output_dir='./results'):
    '''
    Tracking pipeline for single video (experimented with single and multiple tracking but took too long).
    Data is loaded, SAM 3 is run, metrics are calculated, results saved.
    
    Args:
        participant (str): 'P1', 'P2' ...
        trial (int): 1, 2, ...
        mode (str): 'sequential' or 'simultaneous'
        config: DatasetConfig instance (only for testing a single video vs running all of them)
        output_dir: Save results
    
    Returns:
        Tuple of (metrics_dict, predictions_dict, annotations_dict, fps, elapsed_time)
    '''
    
    if config is None:
        config = DatasetConfig()
    
    print(f"Video: {participant}.{trial} (Mode: {mode.upper()})")
    
    # Get paths
    video_path = config.get_video_path(participant, trial)
    csv_path = config.get_csv_path(participant, trial)
    
    if not video_path.exists() or not csv_path.exists():
        print(f"Files not found")
        return None, None, None, None, None
    
    # Setup folders
    trial_name = f"{participant}_{trial}"
    if mode == 'simultaneous':
        trial_name += '_simultaneous'
    
    trial_dir = Path(output_dir) / participant / trial_name
    masks_dir = trial_dir / 'masks'
    vis_dir = trial_dir / 'visualizations'
    
    os.makedirs(trial_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    # Load annotations
    fps = get_video_fps(video_path)
    print(f"Video FPS: {fps}")
    annotations = load_annotations(csv_path, fps)
    
    # Find active tools
    all_tools = set()
    for frame_data in annotations.values():
        for item in frame_data:

            all_tools.add(item['class'])
    
    print(f"Tools: {len(all_tools)}: {sorted(all_tools)}\n")
    
    # Build tool info
    tool_info_dict = {}
    for tool_class in sorted(all_tools):
        tool_frames = sorted([
            f for f, items in annotations.items()
            if any(item['class'] == tool_class for item in items)
        ])
        
        if not tool_frames:
            continue
        
        # The first apperance of the tool with its GT bbox is used for the zero-shot inference
        first_frame = tool_frames[0]
        init_bbox = next(item['bbox'] for item in annotations[first_frame] 
                        if item['class'] == tool_class)
        
        tool_mask_dir = masks_dir / tool_class.replace(' ', '_')
        
        tool_info_dict[tool_class] = {
            'first_frame': tool_frames[0],
            'last_frame': tool_frames[-1],
            'init_bbox': init_bbox,
            'mask_dir': tool_mask_dir
        }
    
    # Track
    tracker = SAM3Tracker()
    
    start_time = time.time()
    
    if mode == 'sequential':
        all_tool_predictions = tracker.track_sequential(video_path, tool_info_dict, chunk_size=100)
    else:  # simultaneous
        all_tool_predictions = tracker.track_simultaneous(video_path, tool_info_dict, chunk_size=100)
    
    elapsed_time = time.time() - start_time
    print(f"\nTracking complete in {elapsed_time:.1f}s")
    
    # Calculate metrics
    print(f"\nMetrics:")
    
    all_predictions = {}
    all_metrics = {}
    
    # Grab metrics from pred to ground truth
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
        
        metrics = calculate_metrics(tool_annotations, tool_predictions)
        all_metrics[tool_class] = metrics
        
        print(f"{tool_class:30s} mAP@50={metrics['mAP@50']:.3f} F1={metrics['f1@50']:.3f}")
        
        # Export CSV
        pred_csv = trial_dir / f"{trial_name.replace('_simultaneous', '')}_{tool_class}_predictions.csv"
        export_predictions_csv(tool_predictions, annotations, fps, pred_csv, tool_class)
    
    # Overall metrics
    overall_metrics = calculate_metrics(annotations, all_predictions)
    all_metrics['OVERALL'] = overall_metrics
    
    print(f"\n{'OVERALL':30s} mAP@50={overall_metrics['mAP@50']:.3f} "
          f"mAP@50-95={overall_metrics['mAP@50-95']:.3f}")
    
    # Export metrics
    metrics_csv = trial_dir / f"{trial_name.replace('_simultaneous', '')}_metrics.csv"
    export_metrics_csv(all_metrics, metrics_csv)
    
    # Generate visualizations
    print(f"\nVISUALIZATIONS")
    
    for tool_class in sorted(all_tools):
        tool_annotations = {
            f: [item for item in items if item['class'] == tool_class]
            for f, items in annotations.items()
            if any(item['class'] == tool_class for item in items)
        }
        
        tool_predictions = all_tool_predictions.get(tool_class, {})
        mask_dir = tool_info_dict[tool_class]['mask_dir']
        
        visualize_tool_masks(tool_class, tool_annotations, tool_predictions, mask_dir, vis_dir)
    
    print(f"\nResults in {trial_dir}")
    
    return all_metrics, all_predictions, annotations, fps, elapsed_time


def process_datasets(participants=None, trials=None, mode='sequential', 
                    output_dir='./results'):
    '''
    Process multiple datasets.
    
    Args:
        participants (list): List like ['P1', 'P2'] or None for all
        trials (dict): Dict like {'P1': [1, 2], 'P2': [1]} or None for [1,2,3,4,5]
        mode (str): 'sequential' or 'simultaneous'
        output_dir (str): Where to save results
    
    Returns:
        Dictionary of all results
    '''
    
    if participants is None:
        participants = ['P1', 'P2', 'P3', 'P4', 'P5']
    
    if trials is None:
        trials = {p: [1, 2, 3, 4, 5] for p in participants}
    
    config = DatasetConfig()
    all_results = {}
    
    for participant in participants:
        for trial in trials.get(participant, [1, 2, 3, 4, 5]):

            dataset_key = f"{participant}.{trial}"
            
            try:
                metrics, predictions, annotations, fps, elapsed = track_video(
                    participant, trial, mode=mode, config=config, output_dir=output_dir
                )
                
                if metrics:
                    all_results[dataset_key] = {
                        'metrics': metrics,
                        'time': elapsed
                    }
                    
            except Exception as e:
                print(f"\nERROR processing {dataset_key}: {e}")
                traceback.print_exc()
    
    # Create summary
    print("ALL DATASETS SUMMARY")
    
    # Writing all dataset summary metrics
    summary_rows = []
    for dataset_key, result in all_results.items():

        overall = result['metrics'].get('OVERALL', {})
        summary_rows.append({
            'dataset': dataset_key,
            'mode': mode,
            'time_sec': f"{result['time']:.1f}",
            'mAP@50': f"{overall.get('mAP@50', 0):.4f}",
            'mAP@75': f"{overall.get('mAP@75', 0):.4f}",
            'mAP@50-95': f"{overall.get('mAP@50-95', 0):.4f}",
            'f1': f"{overall.get('f1@50', 0):.4f}",
        })
    
    summary_csv = Path(output_dir) / f'summary_{mode}.csv'

    with open(summary_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['dataset', 'mode', 'time_sec', 'mAP@50', 
                                               'mAP@75', 'mAP@50-95', 'f1'])
        writer.writeheader()
        writer.writerows(summary_rows)
    
    print(f"Saved to {summary_csv}")
    
    return all_results

def main():
    '''
    Main control flow; processes either a single video or all; track one tool or all at once.
    '''

    # 'sequential' for one tool at once or 'simultaneous' for all
    MODE = 'simultaneous'  
    
    # Single video or all (false for all)
    RUN_SINGLE = False   
    
    # SINGLE VIDEO 
    if RUN_SINGLE:
        track_video(
            participant='P1',
            trial=1,
            mode=MODE,
            output_dir='./results'
        )
    
    # ALL
    else:
        process_datasets(
            participants=['P1', 'P2', 'P3', 'P4', 'P5'],
            trials={'P1': [1, 2, 3, 4, 5],
                   'P2': [1, 2, 3, 4, 5],
                   'P3': [1, 2, 3, 4, 5],
                   'P4': [1, 2, 3, 4, 5],
                   'P5': [1, 2, 3, 4, 5]},
            mode=MODE,
            output_dir='./results'
        )
    
    print(f"Inference complete.")
    

main()