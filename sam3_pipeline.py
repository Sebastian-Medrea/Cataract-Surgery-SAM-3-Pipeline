"""
SAM 3 Video Tracking with bounding box prompts on cataract surgery videos for tool detection

"""

import ast
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import torch
from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
from accelerate import Accelerator


class SAM3VideoTracker:
    """SAM 3 Tracker for bounding box-based video tracking"""
    
    def __init__(self, hf_token=None):
        self.device = Accelerator().device if torch.cuda.is_available() else "cpu"
        print(f"Initializing SAM3Tracker on {self.device}...")
        
        # Use Sam3TrackerVideoModel for bbox prompts
        self.model = Sam3TrackerVideoModel.from_pretrained(
            "facebook/sam3", token=hf_token
        ).to(self.device, dtype=torch.bfloat16)
        
        self.processor = Sam3TrackerVideoProcessor.from_pretrained(
            "facebook/sam3", token=hf_token
        )
        print("SAM3TrackerVideo loaded")
    
    def track_with_bboxes(self, video_path, init_bboxes, start_frame, num_frames):
        """
        Track objects using bounding box prompts
        
        Args:
            video_path: Path to video file
            init_bboxes: List of [xmin, ymin, xmax, ymax] for first frame
            start_frame: Frame where bboxes are valid
            num_frames: Number of frames to track
        
        Returns:
            Dict[frame_idx -> List[{'bbox': [x1,y1,x2,y2], 'score': float, 'obj_id': int}]]
        """
        
        print(f"Tracking {len(init_bboxes)} objects from frame {start_frame}")
        print(f"Video: {Path(video_path).name}")
        
        # Load video frames
        print(f"Loading video frames...")
        video_frames = self._load_frames(video_path, start_frame, num_frames)
        print(f"Loaded {len(video_frames)} frames")
        
        # Initialize video inference session
        print(f"Initializing video session...")
        inference_session = self.processor.init_video_session(
            video=video_frames,
            inference_device=self.device,
            dtype=torch.bfloat16,
        )
        
        # Add bounding box prompts for each object on first frame
        print(f"\nAdding bounding box prompts:")
        obj_ids = list(range(1, len(init_bboxes) + 1))
        
        for obj_id, bbox in zip(obj_ids, init_bboxes):
            print(f"    Object {obj_id}: {bbox}")
            
            # Format is [[bbox]] 
            # [image level, box level, box coordinates]
            input_boxes = [[bbox]]  # 3 levels: [image][boxes][coords]
            
            self.processor.add_inputs_to_inference_session(
                inference_session=inference_session,
                frame_idx=0,  # First frame of loaded sequence
                obj_ids=obj_id,
                input_boxes=input_boxes,
            )
        
        print(f"Added {len(obj_ids)} object prompts")
        
        # CRITICAL: Run inference on first frame to initialize tracking
        print(f"\nInitializing tracking on frame 0...")
        init_output = self.model(
            inference_session=inference_session,
            frame_idx=0,  # Initialize on first frame
        )
        print(f"Tracking initialized")
        
        # Propagate through video
        print(f"\nPropagating through {len(video_frames)} frames...")
        predictions = {}

        # Create folder for saved masks
        os.makedirs('./mask_debug', exist_ok=True)
        
        for sam3_output in self.model.propagate_in_video_iterator(inference_session):
            frame_idx = sam3_output.frame_idx + start_frame
            
            # Post-process masks
            video_res_masks = self.processor.post_process_masks(
                [sam3_output.pred_masks],
                original_sizes=[[inference_session.video_height, inference_session.video_width]],
                binarize=False
            )[0]
            
            # Extract bounding boxes from masks
            frame_predictions = []
            
            for obj_idx, obj_id in enumerate(inference_session.obj_ids):
                if obj_idx < video_res_masks.shape[0]:
                    mask = video_res_masks[obj_idx, 0].float().cpu().numpy()  # [H, W]
                    
                    # Extract bbox from mask
                    rows = np.any(mask > 0.5, axis=1)
                    cols = np.any(mask > 0.5, axis=0)
                    
                    if rows.any() and cols.any():
                        ymin, ymax = np.where(rows)[0][[0, -1]]
                        xmin, xmax = np.where(cols)[0][[0, -1]]
                        
                        # Calculate confidence score
                        score = float(mask.max())
                        
                        frame_predictions.append({
                            'bbox': [int(xmin), int(ymin), int(xmax), int(ymax)],
                            'score': score,
                            'track_id': int(obj_id)
                        })
            
            predictions[frame_idx] = frame_predictions
            
            if frame_idx == start_frame or (frame_idx - start_frame) % 20 == 0:
                print(f"Frame {frame_idx}: {len(frame_predictions)} detections")
        
        print(f"Tracking complete ({len(predictions)} frames)")
        return predictions
    
    def _load_frames(self, video_path, start_frame, num_frames):
        """Load video frames using OpenCV"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frames = []
        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        cap.release()
        return frames


def load_annotations(csv_path, only_tools=True):
    """Load annotations from CSV"""
    annotations = {}
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_idx = int(row['Unnamed: 0'])
            bbox_str = row['Tool bounding box']
            
            if bbox_str == '[]':
                continue
            
            bboxes_raw = ast.literal_eval(bbox_str)
            
            frame_bboxes = []
            for bbox in bboxes_raw:
                tool_class = bbox['class']
                
                if only_tools and tool_class == 'eye':
                    continue
                
                frame_bboxes.append({
                    'bbox': [bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']],
                    'class': tool_class
                })
            
            if frame_bboxes:
                annotations[frame_idx] = frame_bboxes
    
    print(f"Loaded annotations for {len(annotations)} frames (tools only)")
    return annotations


def find_first_tool_frame(annotations):
    """Find first frame with tool annotations"""
    return min(annotations.keys()) if annotations else None


def calculate_iou(bbox1, bbox2):
    """Calculate IoU between two bboxes"""
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


def evaluate_map50(ground_truth, predictions, iou_threshold=0.5):
    """Calculate mAP@50"""
    common_frames = sorted(set(ground_truth.keys()) & set(predictions.keys()))
    
    if not common_frames:
        print("No common frames between GT and predictions!")
        return {'mAP50': 0.0, 'num_frames': 0, 'frame_metrics': []}
    
    print(f"\nEvaluating {len(common_frames)} frames...")
    
    frame_aps = []
    frame_details = []
    
    for frame_idx in common_frames:
        gt_bboxes = [item['bbox'] for item in ground_truth[frame_idx]]
        pred_items = predictions[frame_idx]
        
        if len(gt_bboxes) == 0:
            continue
        
        if len(pred_items) == 0:
            frame_aps.append(0.0)
            frame_details.append({
                'frame': frame_idx,
                'ap': 0.0,
                'gt_count': len(gt_bboxes),
                'pred_count': 0,
                'matched': 0
            })
            continue
        
        pred_items = sorted(pred_items, key=lambda x: x['score'], reverse=True)
        
        true_positives = []
        matched_gt = set()
        
        for pred in pred_items:
            pred_bbox = pred['bbox']
            max_iou = 0
            max_gt_idx = -1
            
            for gt_idx, gt_bbox in enumerate(gt_bboxes):
                if gt_idx in matched_gt:
                    continue
                iou = calculate_iou(pred_bbox, gt_bbox)
                if iou > max_iou:
                    max_iou = iou
                    max_gt_idx = gt_idx
            
            if max_iou >= iou_threshold:
                true_positives.append(1)
                matched_gt.add(max_gt_idx)
            else:
                true_positives.append(0)
        
        tp_cumsum = np.cumsum(true_positives)
        precision = tp_cumsum / np.arange(1, len(true_positives) + 1)
        recall = tp_cumsum / len(gt_bboxes)
        
        ap = 0
        for i in range(len(recall)):
            if i == 0:
                ap += precision[i] * recall[i]
            else:
                ap += precision[i] * (recall[i] - recall[i-1])
        
        frame_aps.append(ap)
        frame_details.append({
            'frame': frame_idx,
            'ap': float(ap),
            'gt_count': len(gt_bboxes),
            'pred_count': len(pred_items),
            'matched': len(matched_gt)
        })
    
    mean_ap = np.mean(frame_aps) if frame_aps else 0.0
    
    return {
        'mAP50': float(mean_ap),
        'num_frames': len(common_frames),
        'frame_metrics': frame_details
    }


def run_experiment(participant, trial, num_frames=100):
    """Run tracking experiment"""
    
    print(f"EXPERIMENT: {participant}.{trial}")
   
    
    # Construct paths
    video_name = f"{participant}.{trial}-1.mkv"
    video_path = Path(f"P:/data/PerkTutor/CataractSurgery/Videos/Simulated_Data/mkv/{video_name}")
    
    if participant == 'P1':
        csv_path = Path(f"P:/data/PerkTutor/CataractSurgery/Datasets/Simulated_Data/P1_{trial}_Revised/{participant}_{trial}_Revised_Labels.csv")
    else:
        csv_path = Path(f"P:/data/PerkTutor/CataractSurgery/Datasets/Simulated_Data/{participant}_{trial}/{participant}_{trial}_Labels.csv")
    
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return None
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return None
    
    print(f"Video: {video_path}")
    print(f"Labels: {csv_path}")
    
    # Load annotations
    annotations = load_annotations(csv_path, only_tools=True)
    
    if not annotations:
        print("No tool annotations found!")
        return None
    
    start_frame = find_first_tool_frame(annotations)
    print(f"First tool appears at frame {start_frame}")
    print(f"Tools: {[item['class'] for item in annotations[start_frame]]}")
    
    init_bboxes = [item['bbox'] for item in annotations[start_frame]]
    
    # Initialize tracker
    tracker = SAM3VideoTracker()
    
    # Run tracking
    predictions = tracker.track_with_bboxes(
        video_path=video_path,
        init_bboxes=init_bboxes,
        start_frame=start_frame,
        num_frames=num_frames
    )
    
    # Evaluate
    results = evaluate_map50(annotations, predictions)
    
    
    print(f"RESULTS: {participant}.{trial}")
    print(f"mAP@50: {results['mAP50']:.3f}")
    print(f"Frames evaluated: {results['num_frames']}")
    
    print(f"\nPer-frame details (first 10):")
    for detail in results['frame_metrics'][:10]:
        print(f"  Frame {detail['frame']:4d}: "
              f"AP={detail['ap']:.3f}, "
              f"GT={detail['gt_count']}, "
              f"Pred={detail['pred_count']}, "
              f"Matched={detail['matched']}")
    
    return results

def visualize_saved_masks(annotations=None, predictions=None):
    """Visualize masks saved during tracking (run after tracking completes)"""
    import glob
    
    mask_files = sorted(glob.glob('./mask_debug/mask_*.npy'))
    
    if not mask_files:
        print("No saved masks found in ./mask_debug/")
        return
    
    print(f"\nVisualizing {len(mask_files)} saved masks...")
    
    for mask_file in mask_files:
        frame_num = int(mask_file.split('_')[-1].split('.')[0])
        
        # Load saved data
        mask = np.load(mask_file)
        frame = np.load(f'./mask_debug/frame_{frame_num:04d}.npy')
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original frame with GT bbox
        axes[0].imshow(frame)
        if annotations and frame_num in annotations:
            gt = annotations[frame_num][0]['bbox']
            axes[0].add_patch(plt.Rectangle((gt[0], gt[1]), gt[2]-gt[0], gt[3]-gt[1],
                              fill=False, edgecolor='lime', linewidth=3))
        axes[0].set_title(f'Frame {frame_num}\n(Green = GT)', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # SAM 3 mask heatmap
        im = axes[1].imshow(mask, cmap='jet', vmin=0, vmax=1)
        axes[1].set_title(f'SAM 3 Mask\nMax: {mask.max():.2f}, Mean: {mask.mean():.2f}', 
                         fontsize=12, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046)
        
        # Mask overlay on frame
        axes[2].imshow(frame)
        mask_overlay = np.zeros((*mask.shape, 4))
        mask_overlay[..., 0] = 1.0  # Red
        mask_overlay[..., 3] = mask * 0.6  # Transparency based on mask value
        axes[2].imshow(mask_overlay)
        
        # Add GT and predicted bboxes
        if annotations and frame_num in annotations:
            gt = annotations[frame_num][0]['bbox']
            axes[2].add_patch(plt.Rectangle((gt[0], gt[1]), gt[2]-gt[0], gt[3]-gt[1],
                              fill=False, edgecolor='lime', linewidth=2, label='GT'))
        
        if predictions and frame_num in predictions and predictions[frame_num]:
            pred = predictions[frame_num][0]['bbox']
            axes[2].add_patch(plt.Rectangle((pred[0], pred[1]), pred[2]-pred[0], pred[3]-pred[1],
                              fill=False, edgecolor='cyan', linewidth=2, linestyle='--', label='Pred'))
            
            # Show IoU
            if annotations and frame_num in annotations:
                iou = calculate_iou(gt, pred)
                axes[2].text(10, 30, f'IoU: {iou:.3f}', color='white', fontsize=14,
                           fontweight='bold', bbox=dict(boxstyle='round',
                           facecolor='green' if iou > 0.5 else 'red', alpha=0.8))
        
        axes[2].set_title('Overlay\n(Red=Mask, Green=GT, Cyan=Pred)', fontsize=12, fontweight='bold')
        axes[2].legend(loc='upper right', fontsize=10)
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'./mask_debug/vis_{frame_num:04d}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Visualized frame {frame_num}")
    
    print(f"\nVisualizations saved to ./mask_debug/vis_*.png")


def main():
    """Run experiment on P1.1"""
    results = run_experiment(
        participant='P1',
        trial=1,
        num_frames=100
    )
    

    # Load data needed for visualization
    csv_path = Path("P:/data/PerkTutor/CataractSurgery/Datasets/Simulated_Data/P1_1_Revised/P1_1_Revised_Labels.csv")
    annotations = load_annotations(csv_path, only_tools=True)
    
    # Get predictions from tracker 
    print("\nGenerating mask visualizations...")
    visualize_saved_masks(annotations=annotations, predictions=None)


main()