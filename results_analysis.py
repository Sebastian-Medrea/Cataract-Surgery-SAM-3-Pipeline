"""
Thesis Results Analysis: SAM 3 Zero-Shot vs YOLOv26 Trained for Cataract Surgery Instrument Tracking.

CISC 499
Student: Sebastian Medrea
Supervisor: Dr. Rebecca Hisey

Compares SAM 3 bounding box tracking (simultaneous + sequential) and YOLOv26 (LOOCV trained)
on the PerkTutor CataractSurgery simulated dataset (25 videos, 5 participants x 5 trials).

Analyses:
- mAP@50 per tool bar chart (SAM3 vs YOLO26)
- mAP@50 boxplots by participant (P1-P5)
- Welch's t-test: SAM3 vs YOLO26 mAP@50
- Motion metrics (path length, usage time) from prediction CSVs
- Spearman correlation: motion metrics vs ICO-OSCAR expert rubric scores (8 criteria)
- Expert vs novice boxplots and Mann-Whitney U tests on motion metrics

Adapted from lab scripts:
- calculate_video_skill_metrics_no_depth.py (bbox center path length)
- PerformStatisticalTests_1.py (t-test/Mann-Whitney, txt output)
- boxplot_skill_groups.py (colour scheme)
- correlation_heatmap.py (heatmap with bold significant values)
- spearman_correlation.py (correlation loop)

Usage: python results_analysis.py
"""

import ast
import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu, spearmanr, ttest_ind

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns


SAM3_DIR = './results'
YOLO_DIR = './yolo26_results'
OUTPUT_DIR = './analysis_output'

# ICO-OSCAR expert evaluation scores
EXPERT_SCORES_CSV = './results_expert_evaluations_1.csv'

PARTICIPANTS = ['P1', 'P2', 'P3', 'P4', 'P5']
TRIALS = [1, 2, 3, 4, 5]
TOOLS = [
    'capsulorhexis forceps',
    'cystotome needle',
    'diamond keratome iso',
    'diamond keratome straight',
    'forceps',
    'viscoelastic cannula'
]
FPS = 15.0

# P1 is the ophthalmologist, P2-P5 are residents.

SKILL_MAP = {
    'P1': 'expert',
    'P2': 'novice',
    'P3': 'novice',
    'P4': 'novice',
    'P5': 'novice'
}

# Colours adapted from lab script: boxplot_skill_groups.py
NOVICE_COLOUR = 'lightskyblue'
EXPERT_COLOUR = 'salmon'
SAM3_COLOUR = '#2ecc71'
YOLO_COLOUR = '#e67e22'

# Plotting params
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12


def sam3_metrics_path(participant, trial, mode):
    '''
    Constructs the path to SAM 3 metrics CSV.
    
    Args:
        participant (str): Participant ID (ex. 'P1')
        trial (int): Trial number (ex. 1)
        mode (str): 'simultaneous' or 'sequential'
        
    Returns:
        Pathlib object mapping to the exact CSV
    '''

    folder = f"{participant}_{trial}"
    if mode == 'simultaneous':
        folder += '_simultaneous'

    return Path(SAM3_DIR) / participant / folder / f"{participant}_{trial}_metrics.csv"


def sam3_pred_path(participant, trial, tool, mode):
    '''
    Constructs the path to SAM 3 per-tool prediction CSV.

    Args:
        participant (str): Participant ID (ex. 'P1')
        trial (int): Trial number (ex. 1)
        tool (str): Tool class name (ex. 'forceps')
        mode (str): 'simultaneous' or 'sequential'
        
    Returns:
        Pathlib object mapping to the per-tool prediction CSV
    '''

    folder = f"{participant}_{trial}"
    if mode == 'simultaneous':
        folder += '_simultaneous'

    return Path(SAM3_DIR) / participant / folder / f"{participant}_{trial}_{tool}_predictions.csv"


def yolo_metrics_path(participant, trial):
    '''
    Constructs the path to YOLOv26 metrics CSV based on LOOCV fold structure.

    Args:
        participant (str): Participant ID (ex. 'P1')
        trial (int): Trial number (ex. 1)
        
    Returns:
        Pathlib object mapping to the metrics CSV
    '''

    return Path(YOLO_DIR) / f"Fold_{participant}" / f"{participant}.{trial}" / f"{participant}.{trial}_metrics.csv"


def yolo_pred_path(participant, trial, tool):
    '''
    Constructs the path to YOLOv26 per-tool prediction CSV.

    Args:
        participant (str): Participant ID (ex. 'P1')
        trial (int): Trial number (ex. 1)
        tool (str): Tool class name (ex. 'forceps')
        
    Returns:
        Pathlib object mapping to the per-tool prediction CSV
    '''

    return Path(YOLO_DIR) / f"Fold_{participant}" / f"{participant}.{trial}" / f"{participant}.{trial}_{tool}_predictions.csv"


def load_expert_scores(csv_path):
    '''
    Load expert evaluation CSV. 
    Video column uses underscores (e.g., P1_1). 
    Returns DataFrame with video_id column matching our P1.1 format for easy merging.
    
    Args:
        csv_path (str): Path to expert scores
        
    Returns:
        pd.DataFrame or None if file missing
    '''

    if csv_path is None or not Path(csv_path).exists():
        print("  Warning: Expert scores CSV not found. Skipping ICO-OSCAR correlation analysis.")
        return None

    try:
        # Handle Byte Order Mark (BOM) common in Excel exports
        scores_df = pd.read_csv(csv_path, encoding='utf-8-sig')  
        scores_df = scores_df.rename(columns={scores_df.columns[0]: 'video_id_raw'})

        # Convert P1_1 to P1.1 format to match prediction tracking data
        scores_df['video_id'] = scores_df['video_id_raw'].str.replace('_', '.', n=1)

        # Clean up column names; extract the number in parentheses as short labels
        criteria_map = {}
        for col in scores_df.columns:
            if col.startswith('('):
                short_label = col.split(')')[0] + ')'  # extracts "(4)"
                criteria_map[col] = short_label

        scores_df = scores_df.rename(columns=criteria_map)
        print(f"Loaded expert scores: {len(scores_df)} videos, criteria: {list(criteria_map.values())}")
        
        return scores_df
        
    except Exception as e:
        print(f"  Error loading expert scores: {e}")
        return None


def load_map_metrics(method, mode='simultaneous'):
    '''
    Load per-tool mAP metrics from _metrics.csv files generated by inference pipelines.
    
    Args:
        method (str): 'sam3' or 'yolo'
        mode (str): 'simultaneous' or 'sequential' (only applicable for SAM 3)
        
    Returns:
        pd.DataFrame compiled with all metrics
    '''

    metrics_data = []
    
    for participant in PARTICIPANTS:
        for trial in TRIALS:
            if method == 'sam3':
                path = sam3_metrics_path(participant, trial, mode)
            else:
                path = yolo_metrics_path(participant, trial)
                
            if not path.exists():
                continue
                
            try:
                df = pd.read_csv(path)
                
                for _, row in df.iterrows():
                    metrics_data.append({
                        'participant': participant, 
                        'trial': trial, 
                        'video_id': f"{participant}.{trial}",
                        'skill_level': SKILL_MAP[participant], 
                        'tool': row['tool'],
                        'mAP50': float(row['mAP@50']), 
                        'mAP75': float(row['mAP@75']),
                        'mAP50_95': float(row['mAP@50-95']),
                        'precision': float(row['precision']), 
                        'recall': float(row['recall']),
                        'f1': float(row['f1']), 
                        'frames': int(row['frames']),
                        'method': f"SAM3_{mode}" if method == 'sam3' else 'YOLO26'
                    })

            except pd.errors.EmptyDataError:
                print(f"Warning: Empty CSV at {path}")
                continue

    result_df = pd.DataFrame(metrics_data)
    print(f"  Loaded {len(result_df)} rows for {method} ({mode if method == 'sam3' else 'trained'})")
    
    return result_df


def parse_bbox(bbox_str):
    '''
    Parse bounding box string to list safely. Handles edge cases like empty, nan, or malformed strings.

    Args:
        bbox_str (str): String representation of bbox (ex. '[10, 20, 100, 200]')

    Returns:
        list or None: [xmin, ymin, xmax, ymax] if valid, None otherwise
    '''

    string_val = str(bbox_str).strip()
    if string_val in ('[]', '', 'nan', 'None'):
        return None
        
    try:
        bbox = ast.literal_eval(string_val)
        if isinstance(bbox, list) and len(bbox) == 4:
            return bbox
    except (ValueError, SyntaxError):
        pass
        
    return None


def calculate_bbox_center(bbox):
    '''
    Calculates the 2D center coordinate of a bounding box.
    Same logic as calculate_video_skill_metrics_no_depth.py.

    Args:
        bbox (list): [xmin, ymin, xmax, ymax]

    Returns:
        tuple: (x_center, y_center)
    '''

    x_center = bbox[0] + (bbox[2] - bbox[0]) / 2.0
    y_center = bbox[1] + (bbox[3] - bbox[1]) / 2.0
    return (x_center, y_center)


def euclidean_distance(point1, point2):
    '''
    Standard Euclidean distance between two points.

    Args:
        point1 (tuple): (x, y)
        point2 (tuple): (x, y)

    Returns:
        float: distance in pixels
    '''
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)


def compute_motion_for_tool(pred_csv_path):
    '''
    Compute center path length (px) and usage time (s) from a per-tool prediction CSV.
    Skips detection gaps > 5 seconds to prevent artificial path length inflation.
    
    Args:
        pred_csv_path (Path): Path to prediction file
        
    Returns:
        dict: Calculated motion metrics with 'path_length', 'usage_time', 'num_detections'
    '''
    empty_metrics = {'path_length': 0.0, 'usage_time': 0.0, 'num_detections': 0}
    
    if not Path(pred_csv_path).exists():
        return empty_metrics
        
    try:
        df = pd.read_csv(pred_csv_path)
    except pd.errors.EmptyDataError:
        return empty_metrics

    detections = []
    for _, row in df.iterrows():
        bbox = parse_bbox(row['pred_bbox'])
        if bbox is not None:
            center = calculate_bbox_center(bbox)
            detections.append({'time': float(row['time']), 'center': center})

    if len(detections) == 0:
        return empty_metrics

    path_length = 0.0
    
    # Calculate path length, skipping massive time jumps
    for i in range(1, len(detections)):
        time_delta = detections[i]['time'] - detections[i-1]['time']
        
        # 5 second threshold from standard lab configuration
        if 0 < time_delta < 5.0:
            path_length += euclidean_distance(detections[i-1]['center'], detections[i]['center'])

    return {
        'path_length': path_length,
        'usage_time': len(detections) / FPS,
        'num_detections': len(detections)
    }


def compute_all_motion_metrics(method, mode='simultaneous'):
    '''
    Computes aggregated motion metrics (path length, usage time) for all participants,
    trials, and tools.

    Args:
        method (str): 'sam3' or 'yolo'
        mode (str): 'simultaneous' or 'sequential' (only applicable for SAM 3)

    Returns:
        pd.DataFrame with path_length, usage_time, and num_detections per tool per video
    '''
    metrics_data = []
    
    for participant in PARTICIPANTS:
        for trial in TRIALS:
            for tool in TOOLS:
                if method == 'sam3':
                    path = sam3_pred_path(participant, trial, tool, mode)
                else:
                    path = yolo_pred_path(participant, trial, tool)
                    
                motion = compute_motion_for_tool(path)
                
                metrics_data.append({
                    'participant': participant, 
                    'trial': trial, 
                    'video_id': f"{participant}.{trial}",
                    'skill_level': SKILL_MAP[participant], 
                    'tool': tool,
                    'path_length': motion['path_length'], 
                    'usage_time': motion['usage_time'],
                    'num_detections': motion['num_detections'],
                    'method': f"SAM3_{mode}" if method == 'sam3' else 'YOLO26'
                })
                
    result_df = pd.DataFrame(metrics_data)
    print(f"  Computed motion metrics: {len(result_df)} rows for {method}")
    
    return result_df


def run_ttest_map50(all_metrics_df, output_path):
    '''
    Performs Welch's t-test (unequal variance) on OVERALL mAP@50.
    Tests SAM3 simultaneous vs YOLO26, SAM3 sequential vs YOLO26,
    and expert vs novice within each method.

    Args:
        all_metrics_df (pd.DataFrame): Combined metrics from all methods
        output_path (str): Path to save results CSV

    Returns:
        pd.DataFrame with t-test results
    '''
    test_results = []
    overall_df = all_metrics_df[all_metrics_df['tool'] == 'OVERALL'].copy()

    # SAM3 variants vs YOLO26
    yolo_values = overall_df[overall_df['method'] == 'YOLO26']['mAP50'].values
    
    for sam3_method in ['SAM3_simultaneous', 'SAM3_sequential']:
        sam3_values = overall_df[overall_df['method'] == sam3_method]['mAP50'].values
        
        if len(sam3_values) >= 2 and len(yolo_values) >= 2:
            stat, p_val = ttest_ind(sam3_values, yolo_values, equal_var=False)
        else:
            stat, p_val = np.nan, np.nan
            
        test_results.append({
            'comparison': f'{sam3_method} vs YOLO26',
            'group_a': sam3_method,
            'group_b': 'YOLO26',
            'n_a': len(sam3_values),
            'n_b': len(yolo_values),
            'mean_a': np.mean(sam3_values) * 100 if len(sam3_values) > 0 else 0,
            'std_a': np.std(sam3_values) * 100 if len(sam3_values) > 0 else 0,
            'mean_b': np.mean(yolo_values) * 100 if len(yolo_values) > 0 else 0,
            'std_b': np.std(yolo_values) * 100 if len(yolo_values) > 0 else 0,
            't_stat': stat,
            'p_value': p_val,
            'significant': p_val < 0.05 if not np.isnan(p_val) else False
        })

    # Expert vs Novice within each method
    for method in overall_df['method'].unique():
        method_df = overall_df[overall_df['method'] == method]
        expert_vals = method_df[method_df['skill_level'] == 'expert']['mAP50'].values
        novice_vals = method_df[method_df['skill_level'] == 'novice']['mAP50'].values
        
        if len(expert_vals) >= 2 and len(novice_vals) >= 2:
            stat, p_val = ttest_ind(expert_vals, novice_vals, equal_var=False)
        else:
            stat, p_val = np.nan, np.nan
            
        test_results.append({
            'comparison': f'{method}: Expert vs Novice',
            'group_a': 'expert',
            'group_b': 'novice',
            'n_a': len(expert_vals),
            'n_b': len(novice_vals),
            'mean_a': np.mean(expert_vals) * 100 if len(expert_vals) > 0 else 0, 
            'std_a': np.std(expert_vals) * 100 if len(expert_vals) > 0 else 0,
            'mean_b': np.mean(novice_vals) * 100 if len(novice_vals) > 0 else 0, 
            'std_b': np.std(novice_vals) * 100 if len(novice_vals) > 0 else 0,
            't_stat': stat,
            'p_value': p_val,
            'significant': p_val < 0.05 if not np.isnan(p_val) else False
        })

    # Export structured CSV
    results_df = pd.DataFrame(test_results)
    results_df.to_csv(output_path, index=False)

    # Export t-tests 
    txt_path = output_path.replace('.csv', '.txt')
    with open(txt_path, 'w') as f:
        f.write("Welch's t-test on mAP@50\n")
        f.write("=" * 80 + "\n\n")
        
        for _, row in results_df.iterrows():
            significance_flag = " ***" if row['significant'] else ""
            f.write(f"{row['comparison']}\n")
            f.write(f"  {row['group_a']:20s} (n={row['n_a']}): {row['mean_a']:.2f}% +/- {row['std_a']:.2f}%\n")
            f.write(f"  {row['group_b']:20s} (n={row['n_b']}): {row['mean_b']:.2f}% +/- {row['std_b']:.2f}%\n")
            f.write(f"  t={row['t_stat']:.3f}, p={row['p_value']:.5f}{significance_flag}\n\n")

    print(f"Saved t-tests to {output_path}")
    return results_df


def run_mannwhitney_motion(motion_df, output_path):
    '''
    Performs Mann-Whitney U test: expert (P1) vs novice (P2-P5) on motion metrics.
    Done separately for total motion and per-tool motion.

    Args:
        motion_df (pd.DataFrame): Motion metrics for all methods
        output_path (str): Path to save results CSV

    Returns:
        pd.DataFrame with Mann-Whitney results
    '''
    test_results = []
    
    for method in motion_df['method'].unique():
        method_df = motion_df[motion_df['method'] == method]

        # Aggregate total motion per video
        video_totals = method_df.groupby(['video_id', 'skill_level']).agg(
            total_path_length=('path_length', 'sum'),
            total_usage_time=('usage_time', 'sum')
        ).reset_index()

        # Compare overall totals
        for metric_col in ['total_path_length', 'total_usage_time']:
            expert_vals = video_totals[video_totals['skill_level'] == 'expert'][metric_col].values
            novice_vals = video_totals[video_totals['skill_level'] == 'novice'][metric_col].values
            
            if len(expert_vals) > 0 and len(novice_vals) > 0:
                stat, p_val = mannwhitneyu(expert_vals, novice_vals, alternative='two-sided')
            else:
                stat, p_val = np.nan, np.nan
                
            test_results.append({
                'method': method,
                'tool': 'ALL_TOOLS',
                'metric': metric_col,
                'expert_mean': np.mean(expert_vals),
                'expert_std': np.std(expert_vals),
                'novice_mean': np.mean(novice_vals),
                'novice_std': np.std(novice_vals),
                'U_stat': stat,
                'p_value': p_val,
                'significant': p_val < 0.05 if not np.isnan(p_val) else False
            })

        # Compare individual tools
        for tool in TOOLS:
            tool_df = method_df[method_df['tool'] == tool]
            
            for metric_col in ['path_length', 'usage_time']:
                expert_vals = tool_df[tool_df['skill_level'] == 'expert'][metric_col].values
                novice_vals = tool_df[tool_df['skill_level'] == 'novice'][metric_col].values
                
                # Ensure data arrays contain non-zero variance before testing
                if len(expert_vals) > 0 and len(novice_vals) > 0 and (np.any(expert_vals) or np.any(novice_vals)):
                    try:
                        stat, p_val = mannwhitneyu(expert_vals, novice_vals, alternative='two-sided')
                    except ValueError:
                        stat, p_val = np.nan, np.nan
                else:
                    stat, p_val = np.nan, np.nan
                    
                test_results.append({
                    'method': method,
                    'tool': tool,
                    'metric': metric_col,
                    'expert_mean': np.mean(expert_vals) if len(expert_vals) > 0 else 0,
                    'expert_std': np.std(expert_vals) if len(expert_vals) > 0 else 0,
                    'novice_mean': np.mean(novice_vals) if len(novice_vals) > 0 else 0,
                    'novice_std': np.std(novice_vals) if len(novice_vals) > 0 else 0,
                    'U_stat': stat,
                    'p_value': p_val,
                    'significant': p_val < 0.05 if not np.isnan(p_val) else False
                })

    # Export CSV
    results_df = pd.DataFrame(test_results)
    results_df.to_csv(output_path, index=False)

    # Export text file
    txt_path = output_path.replace('.csv', '.txt')
    with open(txt_path, 'w') as f:
        f.write("Mann-Whitney U: Expert (P1) vs Novice (P2-P5) on Motion Metrics\n")
        f.write("=" * 80 + "\n")
        
        for method in results_df['method'].unique():
            f.write(f"\n{method}\n\n")
            method_subset = results_df[results_df['method'] == method]
            
            for _, row in method_subset.iterrows():
                significance_flag = " *" if row['significant'] else ""
                f.write(f"  {row['tool']:30s} {row['metric']:25s} "
                        f"Exp={row['expert_mean']:8.1f}+/-{row['expert_std']:6.1f}  "
                        f"Nov={row['novice_mean']:8.1f}+/-{row['novice_std']:6.1f}  "
                        f"p={row['p_value']:.4f}{significance_flag}\n")

    print(f"Saved Mann-Whitney to {output_path}")
    return results_df


def run_spearman_vs_expert_scores(motion_df, expert_scores_df, output_path):
    '''
    Spearman correlation between per-tool motion metrics and each ICO-OSCAR criterion.
    This logic mirrors the analysis framework of the original IJCARS paper.

    Args:
        motion_df (pd.DataFrame): Motion metrics for all methods
        expert_scores_df (pd.DataFrame): ICO-OSCAR expert evaluation scores
        output_path (str): Path to save results CSV

    Returns:
        pd.DataFrame with Spearman results, or None if no expert scores
    '''
    if expert_scores_df is None:
        print("  No expert scores provided, skipping Spearman evaluation.")
        return None

    # Filter out the criterion columns (they all begin with a parenthesis)
    criteria_columns = [col for col in expert_scores_df.columns if col.startswith('(')]
    test_results = []
    
    for method in motion_df['method'].unique():
        method_df = motion_df[motion_df['method'] == method]

        # Per-tool correlations
        for tool_class in TOOLS:
            tool_df = method_df[method_df['tool'] == tool_class][['video_id', 'path_length', 'usage_time']].copy()
            merged_df = tool_df.merge(expert_scores_df[['video_id'] + criteria_columns], on='video_id', how='inner')

            for metric_col in ['path_length', 'usage_time']:
                for criterion in criteria_columns:
                    predicted_vals = merged_df[metric_col].values
                    score_values = merged_df[criterion].values
                    
                    # Spearman requires variance to run accurately
                    if len(predicted_vals) > 2 and np.std(predicted_vals) > 0 and np.std(score_values) > 0:
                        rho, p_val = spearmanr(predicted_vals, score_values)
                    else:
                        rho, p_val = 0.0, 1.0
                        
                    test_results.append({
                        'method': method,
                        'tool': tool_class,
                        'metric': metric_col,
                        'criterion': criterion,
                        'spearman_rho': rho,
                        'p_value': p_val,
                        'significant': p_val < 0.05
                    })

        # Total aggregated correlations across all tools
        video_totals = method_df.groupby('video_id').agg(
            total_path_length=('path_length', 'sum'),
            total_usage_time=('usage_time', 'sum')
        ).reset_index()
        
        merged_total_df = video_totals.merge(expert_scores_df[['video_id'] + criteria_columns], on='video_id', how='inner')

        for metric_col in ['total_path_length', 'total_usage_time']:
            for criterion in criteria_columns:
                predicted_vals = merged_total_df[metric_col].values
                score_values = merged_total_df[criterion].values
                
                if len(predicted_vals) > 2 and np.std(predicted_vals) > 0 and np.std(score_values) > 0:
                    rho, p_val = spearmanr(predicted_vals, score_values)
                else:
                    rho, p_val = 0.0, 1.0
                    
                test_results.append({
                    'method': method,
                    'tool': 'ALL_TOOLS',
                    'metric': metric_col,
                    'criterion': criterion,
                    'spearman_rho': rho,
                    'p_value': p_val,
                    'significant': p_val < 0.05
                })

    results_df = pd.DataFrame(test_results)
    results_df.to_csv(output_path, index=False)

    # Console summary of significance
    for method in results_df['method'].unique():
        method_subset = results_df[results_df['method'] == method]
        num_significant = method_subset['significant'].sum()
        total_tests = len(method_subset)
        print(f"  {method}: {num_significant}/{total_tests} significant correlations")

    return results_df


# Full display names for tools 
TOOL_DISPLAY = {
    'capsulorhexis forceps': 'Capsulorhexis forceps',
    'cystotome needle': 'Cystotome needle',
    'diamond keratome iso': 'Diamond keratome (angled)',
    'diamond keratome straight': 'Diamond keratome (straight)',
    'forceps': 'Forceps',
    'viscoelastic cannula': 'Viscoelastic cannula',
    'ALL_TOOLS': 'Total (all tools)'
}

CRITERION_DISPLAY = {
    '(4)': '(4)',
    '(5)': '(5)',
    '(6)': '(6)',
    '(7)': '(7)',
    '(14)': '(14)',
    '(15)': '(15)',
    '(16)': '(16)',
    '(17)': '(17)'
}

METHOD_DISPLAY = {
    'SAM3_simultaneous': 'SAM 3 Simultaneous',
    'SAM3_sequential': 'SAM 3 Sequential',
    'YOLO26': 'YOLOv26'
}


def set_box_colors(boxplot_dict, colors):
    '''
    Colors individual boxplots dynamically based on skill level.
    Adapted from Compare_novice_metrics_to_experts.py.
    '''

    for patch, color in zip(boxplot_dict['boxes'], colors):
        patch.set_facecolor(color)
        
    for median in boxplot_dict['medians']:
        plt.setp(median, color='black')


def plot_map50_per_tool(sam3_df, yolo_df, save_path):
    '''
    Grouped bar chart comparing mean mAP@50 per tool.
    Uncertainty/variance is handled in the participant boxplot function.

    Args:
        sam3_df (pd.DataFrame): SAM 3 metrics
        yolo_df (pd.DataFrame): YOLO metrics
        save_path (str): Output file path for PNG
    '''

    sam3_tools = sam3_df[sam3_df['tool'] != 'OVERALL']
    yolo_tools = yolo_df[yolo_df['tool'] != 'OVERALL']

    sam3_avg = sam3_tools.groupby('tool')['mAP50'].mean()
    yolo_avg = yolo_tools.groupby('tool')['mAP50'].mean()

    common_tools = sorted(set(sam3_avg.index) & set(yolo_avg.index))
    if not common_tools:
        return

    x_positions = np.arange(len(common_tools))
    bar_width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))

    sam3_values = [sam3_avg.get(t, 0) * 100 for t in common_tools]
    yolo_values = [yolo_avg.get(t, 0) * 100 for t in common_tools]

    bar1 = ax.bar(x_positions - bar_width/2, sam3_values, bar_width, label='SAM 3 (zero-shot)',
                  color=SAM3_COLOUR, edgecolor='black', linewidth=0.5)
    bar2 = ax.bar(x_positions + bar_width/2, yolo_values, bar_width, label='YOLOv26 (trained)',
                  color=YOLO_COLOUR, edgecolor='black', linewidth=0.5)

    # Annotate value percentages directly above bars
    for bar in list(bar1) + list(bar2):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 1.5, f'{height:.1f}%',
                ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([TOOL_DISPLAY.get(t, t) for t in common_tools],
                       fontsize=11, rotation=30, ha='right')
    ax.set_ylabel('Mean mAP@50 (%)', fontsize=13)
    ax.set_ylim(0, 105)
    ax.set_title('Per-Instrument Detection Accuracy: SAM 3 Zero-Shot vs. Trained YOLOv26',
                 fontsize=14, pad=12)
                 
    ax.legend(fontsize=12, frameon=False)
    ax.yaxis.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot: {save_path}")


def plot_map50_by_participant(sam3_df, yolo_df, save_path):
    '''
    Boxplot of overall mAP@50 performance variance per participant.

    Args:
        sam3_df (pd.DataFrame): SAM 3 metrics
        yolo_df (pd.DataFrame): YOLO metrics
        save_path (str): Output file path for PNG
    '''

    sam3_overall = sam3_df[sam3_df['tool'] == 'OVERALL'].copy()
    yolo_overall = yolo_df[yolo_df['tool'] == 'OVERALL'].copy()
    
    sam3_overall['pct'] = sam3_overall['mAP50'] * 100
    yolo_overall['pct'] = yolo_overall['mAP50'] * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    offset = 0.2
    width = 0.35
    box_properties = dict(linewidth=1.5)

    # Isolate data for plotting
    sam3_data = []
    yolo_data = []

    for p in PARTICIPANTS:
        sam3_data.append(sam3_overall[sam3_overall['participant'] == p]['pct'].values)
        yolo_data.append(yolo_overall[yolo_overall['participant'] == p]['pct'].values)

    # Draw boxplots
    sam3_boxplot = ax.boxplot(sam3_data, positions=[i*2 - offset for i in range(5)], widths=width,
                              patch_artist=True, showfliers=True, boxprops=box_properties,
                              whiskerprops=box_properties, capprops=box_properties,
                              medianprops=dict(color='black', linewidth=2))
                              
    yolo_boxplot = ax.boxplot(yolo_data, positions=[i*2 + offset for i in range(5)], widths=width,
                              patch_artist=True, showfliers=True, boxprops=box_properties,
                              whiskerprops=box_properties, capprops=box_properties,
                              medianprops=dict(color='black', linewidth=2))

    # Apply specific coloring
    for box in sam3_boxplot['boxes']:
        box.set(facecolor=SAM3_COLOUR, edgecolor='black')

    for box in yolo_boxplot['boxes']:
        box.set(facecolor=YOLO_COLOUR, edgecolor='black')

    ax.set_xticks([i*2 for i in range(5)])
    ax.set_xticklabels(PARTICIPANTS, fontsize=12)
    ax.set_ylabel('Overall mAP@50 (%)', fontsize=13)
    ax.set_title('Cross-Participant Generalization of Detection Performance', fontsize=14, pad=12)
    ax.yaxis.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.legend(handles=[
        Patch(facecolor=SAM3_COLOUR, edgecolor='black', label='SAM 3 (zero-shot)'),
        Patch(facecolor=YOLO_COLOUR, edgecolor='black', label='YOLOv26 (trained)')
    ], fontsize=12, frameon=False)
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot: {save_path}")


def plot_expert_vs_novice(motion_df, metric_col, ylabel, save_path):
    '''
    Side-by-side boxplots comparing expert vs novice motion metrics per tool.
    Generates one subplot per method (SAM3, YOLO).

    Args:
        motion_df (pd.DataFrame): Motion metrics for all methods
        metric_col (str): Column name to plot ('path_length' or 'usage_time')
        ylabel (str): Y-axis label
        save_path (str): Output file path for PNG
    '''

    methods = sorted(motion_df['method'].unique())
    num_methods = len(methods)
    
    fig, axes = plt.subplots(1, num_methods, figsize=(6.5 * num_methods, 6), sharey=False)
    if num_methods == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        method_df = motion_df[motion_df['method'] == method]
        offset = 0.2
        width = 0.35
        box_props = dict(linewidth=1.5)

        novice_data = []
        expert_data = []
        for tool in TOOLS:
            tool_subset = method_df[method_df['tool'] == tool]
            novice_data.append(tool_subset[tool_subset['skill_level'] == 'novice'][metric_col].values)
            expert_data.append(tool_subset[tool_subset['skill_level'] == 'expert'][metric_col].values)

        boxplot_novice = ax.boxplot(novice_data, positions=[i*2 - offset for i in range(len(TOOLS))],
                                    widths=width, patch_artist=True, showfliers=False,
                                    boxprops=box_props, whiskerprops=box_props, capprops=box_props,
                                    medianprops=dict(color='black', linewidth=2))
                                    
        boxplot_expert = ax.boxplot(expert_data, positions=[i*2 + offset for i in range(len(TOOLS))],
                                    widths=width, patch_artist=True, showfliers=False,
                                    boxprops=box_props, whiskerprops=box_props, capprops=box_props,
                                    medianprops=dict(color='black', linewidth=2))

        set_box_colors(boxplot_novice, [NOVICE_COLOUR] * len(TOOLS))
        set_box_colors(boxplot_expert, [EXPERT_COLOUR] * len(TOOLS))

        ax.set_xticks([i*2 for i in range(len(TOOLS))])
        ax.set_xticklabels([TOOL_DISPLAY.get(t, t) for t in TOOLS], fontsize=9, rotation=35, ha='right')
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(METHOD_DISPLAY.get(method, method), fontsize=13)
        ax.yaxis.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[-1].legend(handles=[
        Patch(facecolor=NOVICE_COLOUR, edgecolor='black', label='Novice (P2-P5)'),
        Patch(facecolor=EXPERT_COLOUR, edgecolor='black', label='Expert (P1)')
    ], fontsize=10, frameon=False, loc='upper right')

    metric_label = 'Path Length' if 'path' in metric_col else 'Usage Time'
    fig.suptitle(f'Expert vs. Novice {metric_label} per Instrument Across Detection Methods', fontsize=14, y=1.02)
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {save_path}")


def plot_ico_heatmap(spearman_df, save_path_prefix):
    '''
    Generates Spearman correlation heatmaps for path length and usage time.
    Significant correlations (p < 0.05) are bolded inside the heatmap.

    Args:
        spearman_df (pd.DataFrame): Spearman results from run_spearman_vs_expert_scores
        save_path_prefix (str): Output file path prefix (method name is appended)
    '''
    if spearman_df is None or len(spearman_df) == 0:
        return

    tool_order = [
        'forceps', 'diamond keratome straight', 'viscoelastic cannula',
        'cystotome needle', 'diamond keratome iso', 'capsulorhexis forceps',
        'ALL_TOOLS'
    ]
    crit_order = ['(4)', '(5)', '(6)', '(7)', '(14)', '(15)', '(16)', '(17)']

    for method in spearman_df['method'].unique():
        method_df = spearman_df[spearman_df['method'] == method].copy()
        display_method = METHOD_DISPLAY.get(method, method)

        metric_pairs = [
            ('path_length', 'total_path_length', 'Path length'),
            ('usage_time', 'total_usage_time', 'Usage time')
        ]

        fig, axes = plt.subplots(1, 2, figsize=(18, 5.5))

        for ax, (per_tool_metric, total_metric, metric_label) in zip(axes, metric_pairs):
            row_labels = []
            matrix_vals = []
            matrix_sig = []

            # Populate the heatmap data array
            for tool in tool_order:
                metric_filter = total_metric if tool == 'ALL_TOOLS' else per_tool_metric
                tool_data = method_df[(method_df['tool'] == tool) & (method_df['metric'] == metric_filter)]

                row_vals = []
                row_sig = []

                for criterion in crit_order:
                    matching_row = tool_data[tool_data['criterion'] == criterion]

                    if len(matching_row) > 0:
                        row_vals.append(matching_row.iloc[0]['spearman_rho'])
                        row_sig.append(matching_row.iloc[0]['significant'])

                    else:
                        row_vals.append(0.0)
                        row_sig.append(False)

                row_labels.append(TOOL_DISPLAY.get(tool, tool))
                matrix_vals.append(row_vals)
                matrix_sig.append(row_sig)

            matrix_vals = np.array(matrix_vals)
            matrix_sig = np.array(matrix_sig)

            # Map the text annotations, bold if significant
            annotated_text = np.empty_like(matrix_vals, dtype=object)
            for i in range(matrix_vals.shape[0]):
                for j in range(matrix_vals.shape[1]):

                    correlation_value = matrix_vals[i, j]

                    if matrix_sig[i, j]:
                        annotated_text[i, j] = f"$\\bf{{{correlation_value:.2f}}}$"
                    else:
                        annotated_text[i, j] = f"{correlation_value:.2f}"

            col_labels = [CRITERION_DISPLAY.get(c, c) for c in crit_order]

            sns.heatmap(matrix_vals, annot=annotated_text, fmt='', cmap='coolwarm',
                        vmin=-0.8, vmax=0.8, center=0,
                        xticklabels=col_labels, yticklabels=row_labels,
                        linewidths=0.5, linecolor='black',
                        annot_kws={'size': 13, 'color': 'black'},
                        cbar_kws={'label': 'Correlation Strength', 'shrink': 0.9},
                        ax=ax)

            ax.set_title(metric_label, fontsize=14)
            ax.set_xlabel('Scoring criteria', fontsize=12)
            ax.set_ylabel('', fontsize=12)
            ax.tick_params(axis='x', labelsize=12, rotation=0)
            ax.tick_params(axis='y', labelsize=11, rotation=0)

        fig.suptitle(f'{display_method}: Spearman Rank Correlations Between Instrument Motion Metrics and ICO-OSCAR Expert Evaluation Scores',
                     fontsize=14, y=1.01)
                     
        plt.tight_layout()

        safe_method_name = method.replace(' ', '_')
        out_path = f"{save_path_prefix}_{safe_method_name}.png"
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved plot: {out_path}")


def plot_ttest_comparison(all_metrics_df, ttest_csv_path, save_path):
    '''
    Generates a bar chart of mean overall mAP@50 per method with standard error bars.
    Overlays significance brackets with p-values extracted from the Welch's t-test CSV.

    Args:
        all_metrics_df (pd.DataFrame): Combined metrics from all methods
        ttest_csv_path (str): Path to t-test results CSV
        save_path (str): Output file path for PNG
    '''
    overall_df = all_metrics_df[all_metrics_df['tool'] == 'OVERALL'].copy()
    overall_df['mAP50_pct'] = overall_df['mAP50'] * 100

    ttest_df = pd.read_csv(ttest_csv_path)

    methods_to_plot = ['SAM3_simultaneous', 'SAM3_sequential', 'YOLO26']
    display_names = {
        'SAM3_simultaneous': 'SAM 3\nSimultaneous',
        'SAM3_sequential': 'SAM 3\nSequential',
        'YOLO26': 'YOLOv26\n(trained)'
    }
    colors_map = {
        'SAM3_simultaneous': SAM3_COLOUR,
        'SAM3_sequential': '#27ae60',  # Darker green for sequential
        'YOLO26': YOLO_COLOUR
    }

    means = []
    stds = []
    labels = []
    bar_colors = []
    present_methods = []
    
    for method in methods_to_plot:
        subset = overall_df[overall_df['method'] == method]['mAP50_pct']
        if len(subset) == 0:
            continue
            
        means.append(subset.mean())
        stds.append(subset.std())
        labels.append(display_names[method])
        bar_colors.append(colors_map[method])
        present_methods.append(method)

    x_positions = np.arange(len(means))
    fig, ax = plt.subplots(figsize=(8, 6))

    bars = ax.bar(x_positions, means, yerr=stds, capsize=6, color=bar_colors,
                  edgecolor='black', linewidth=0.8, width=0.55,
                  error_kw={'linewidth': 1.2})

    # Overlay mean +/- std strings
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + stds[i] + 2,
                f'{means[i]:.1f} +/- {stds[i]:.1f}%',
                ha='center', va='bottom', fontsize=10)

    # Starting height for significance brackets
    y_top = max([m + s for m, s in zip(means, stds)]) + 12

    for sam3_method in ['SAM3_simultaneous', 'SAM3_sequential']:
        if sam3_method not in present_methods or 'YOLO26' not in present_methods:
            continue

        match = ttest_df[ttest_df['comparison'].str.contains(sam3_method) &
                         ttest_df['comparison'].str.contains('YOLO26')]
        if len(match) == 0:
            continue

        p_val = match.iloc[0]['p_value']
        idx_a = present_methods.index(sam3_method)
        idx_b = present_methods.index('YOLO26')

        p_text = 'p < 0.001' if p_val < 0.001 else f'p = {p_val:.3f}'

        # Draw structural bracket
        ax.plot([idx_a, idx_a, idx_b, idx_b],
                [y_top - 1, y_top, y_top, y_top - 1],
                color='black', linewidth=1.2)
        ax.text((idx_a + idx_b) / 2, y_top + 0.5, p_text,
                ha='center', va='bottom', fontsize=11)
                
        y_top += 10  # Shift up for subsequent brackets

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('Overall mAP@50 (%)', fontsize=13)
    ax.set_title("Welch's t-test: Overall Detection Performance by Method", fontsize=14, pad=12)
    ax.set_ylim(0, y_top + 10)
    ax.yaxis.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot: {save_path}")


def save_summary_table(all_metrics_df, save_path):
    '''
    Compiles average OVERALL mAP@50 metrics per participant.

    Args:
        all_metrics_df (pd.DataFrame): Combined metrics from all methods
        save_path (str): Output file path for CSV
    '''
    overall_df = all_metrics_df[all_metrics_df['tool'] == 'OVERALL'].copy()
    
    summary_df = overall_df.groupby(['participant', 'skill_level', 'method']).agg(
        mean_mAP50=('mAP50', 'mean'), 
        std_mAP50=('mAP50', 'std'),
        mean_f1=('f1', 'mean'), 
        n_trials=('mAP50', 'count')
    ).reset_index()
    
    summary_df['mean_mAP50_pct'] = (summary_df['mean_mAP50'] * 100).round(2)
    summary_df['std_mAP50_pct'] = (summary_df['std_mAP50'] * 100).round(2)
    
    summary_df.to_csv(save_path, index=False)
    print(f"  Saved summary table: {save_path}")


def main():
    '''
    Main control flow. Coordinates data loading, statistical test execution, 
    and visualization generation.
    '''
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load expert scores
    print("LOADING EXPERT SCORES")
    expert_scores = load_expert_scores(EXPERT_SCORES_CSV)

    # Compile bounding box performance
    print("\nLOADING mAP METRICS")
    sam3_sim_metrics = load_map_metrics('sam3', 'simultaneous')
    sam3_seq_metrics = load_map_metrics('sam3', 'sequential')
    yolo_metrics = load_map_metrics('yolo')
    
    all_metrics = pd.concat([sam3_sim_metrics, sam3_seq_metrics, yolo_metrics], ignore_index=True)
    all_metrics.to_csv(os.path.join(OUTPUT_DIR, 'all_map_metrics.csv'), index=False)

    for method in all_metrics['method'].unique():
        method_subset = all_metrics[(all_metrics['method'] == method) & (all_metrics['tool'] == 'OVERALL')]
        if len(method_subset) > 0:
            print(f"  {method}: {len(method_subset)} videos processed | Avg mAP@50 = {method_subset['mAP50'].mean() * 100:.1f}%")

    # Calculate path length and usage time
    print("\nCOMPUTING MOTION METRICS")
    motion_sim = compute_all_motion_metrics('sam3', 'simultaneous')
    motion_seq = compute_all_motion_metrics('sam3', 'sequential')
    motion_yolo = compute_all_motion_metrics('yolo')
    
    all_motion = pd.concat([motion_sim, motion_seq, motion_yolo], ignore_index=True)
    all_motion.to_csv(os.path.join(OUTPUT_DIR, 'motion_metrics_all.csv'), index=False)

    # Inferential statistical testing
    print("\nEXECUTING STATISTICAL TESTS")
    run_ttest_map50(all_metrics, os.path.join(OUTPUT_DIR, 'ttest_map50.csv'))
    run_mannwhitney_motion(all_motion, os.path.join(OUTPUT_DIR, 'mannwhitney_motion.csv'))

    # OSCAR correlative assessment
    print("\nEVALUATING SPEARMAN RHO VS ICO-OSCAR SCORES")
    spearman_ico_df = run_spearman_vs_expert_scores(all_motion, expert_scores, os.path.join(OUTPUT_DIR, 'spearman_vs_ico_scores.csv'))

    # Data Visualization
    print("\nGENERATING VISUAL PLOTS")
    sam3_primary = sam3_sim_metrics if len(sam3_sim_metrics) > 0 else sam3_seq_metrics
    print(f"Primary SAM 3 baseline set to: {'simultaneous' if len(sam3_sim_metrics) > 0 else 'sequential'}")

    if len(sam3_primary) > 0 and len(yolo_metrics) > 0:
        plot_map50_per_tool(sam3_primary, yolo_metrics, os.path.join(OUTPUT_DIR, 'bar_chart_map50_per_tool.png'))
        plot_map50_by_participant(sam3_primary, yolo_metrics, os.path.join(OUTPUT_DIR, 'boxplot_map50_by_participant.png'))

    if len(all_motion) > 0:
        plot_expert_vs_novice(all_motion, 'path_length', 'Path length (pixels)', os.path.join(OUTPUT_DIR, 'boxplot_path_length.png'))
        plot_expert_vs_novice(all_motion, 'usage_time', 'Usage time (seconds)', os.path.join(OUTPUT_DIR, 'boxplot_usage_time.png'))

    plot_ico_heatmap(spearman_ico_df, os.path.join(OUTPUT_DIR, 'heatmap_ico'))

    ttest_csv = os.path.join(OUTPUT_DIR, 'ttest_map50.csv')
    if len(all_metrics) > 0 and os.path.exists(ttest_csv):
        plot_ttest_comparison(all_metrics, ttest_csv, os.path.join(OUTPUT_DIR, 'ttest_method_comparison.png'))

    save_summary_table(all_metrics, os.path.join(OUTPUT_DIR, 'summary_table.csv'))
    print(f"\nExecution Complete. All exported to {OUTPUT_DIR}")

main()
