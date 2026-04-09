"""
Main Entry Point for the Cataract Surgery Tool Tracking Pipeline.

CISC 499
Student: Sebastian Medrea
Supervisor: Dr. Rebecca Hisey

Provides a command-line interface to run individual pipelines or the full workflow.

Usage:
    python run.py sam3                          Run SAM 3 zero-shot tracking (all participants)
    python run.py yolo                          Run YOLO GT-trained LOOCV
    python run.py yolo-sam3                     Run YOLO trained on SAM 3 pseudo-labels
    python run.py analyze                       Run statistical analysis and plotting
    python run.py all                           Run everything in order

    python run.py sam3 --config my_config.yaml  Use a custom config file
    python run.py sam3 --single                 Quick test on one video only
    python run.py yolo --participants P1 P2 --epochs 50
    python run.py analyze --output ./my_results
"""

import argparse
import sys

from utils import load_config


def run_sam3(config, args):
    '''
    Run the SAM 3 zero-shot tracking pipeline.
    Supports both sequential (one tool at a time) and simultaneous (all tools in one pass) modes.
    '''

    from sam3_pipeline import process_datasets

    # CLI args override config defaults for testing
    mode = args.mode if args.mode else config['sam3']['mode']
    participants = args.participants if args.participants else config['participants']
    output_dir = args.output if args.output else config['output']['sam3_results']

    trials = {p: config['trials_per_participant'] for p in participants}

    if args.single:

        # Run on just one video for quick testing 
        from sam3_pipeline import track_video
        track_video(
            participant=participants[0],
            trial=config['trials_per_participant'][0],
            mode=mode,
            output_dir=output_dir
        )
    else:
        process_datasets(
            participants=participants,
            trials=trials,
            mode=mode,
            output_dir=output_dir
        )


def run_yolo(config, args):
    '''
    Run the YOLOv26 LOOCV pipeline trained on ground truth bounding boxes.
    Each fold holds out one participant for testing and trains on the other four.
    '''

    from yolo26_pipeline import process_datasets_yolo

    participants = args.participants if args.participants else config['participants']
    output_dir = args.output if args.output else config['output']['yolo_results']
    model_path = args.model if args.model else config['yolo']['model_path']
    epochs = args.epochs if args.epochs else config['yolo']['epochs']

    process_datasets_yolo(
        participants=participants,
        output_dir=output_dir,
        model_path=model_path,
        epochs=epochs
    )


def run_yolo_sam3(config, args):
    '''
    Run the YOLOv26 LOOCV pipeline trained on SAM 3 pseudo-labels instead of ground truth.
    Tests whether SAM 3 zero-shot predictions are good enough to replace manual annotations.
    '''

    from yolo26_sam3_combined import process_sam3_to_yolo

    participants = args.participants if args.participants else config['participants']
    output_dir = args.output if args.output else config['output']['yolo_sam3_results']
    model_path = args.model if args.model else config['yolo']['model_path']
    epochs = args.epochs if args.epochs else config['yolo']['epochs']

    process_sam3_to_yolo(
        participants=participants,
        output_dir=output_dir,
        model_path=model_path,
        epochs=epochs
    )


def run_analyze(config, args):
    '''
    Run statistical analysis and generate all figures.
    Compares SAM 3 vs YOLO detection accuracy, computes motion metrics,
    and correlates with ICO-OSCAR expert evaluation scores.
    '''

    from results_analysis import main as analysis_main

    output_dir = args.output if args.output else config['output']['analysis_output']

    analysis_main(config_path=args.config, output_dir=output_dir)


def main():
    '''
    Parse CLI arguments for the selected pipeline.
    All pipelines read from config.yaml by default; CLI flags override specific settings.
    '''

    parser = argparse.ArgumentParser(
        description="Cataract Surgery Tool Tracking: SAM 3 vs YOLOv26",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py sam3                          Run SAM 3 tracking (all participants)
  python run.py sam3 --single                 Quick test on first video only
  python run.py sam3 --mode sequential        Override tracking mode
  python run.py yolo --epochs 50              Train YOLO with 50 epochs
  python run.py yolo --participants P1 P2     Only run on P1 and P2
  python run.py yolo-sam3                     Train YOLO on SAM 3 pseudo-labels
  python run.py analyze --output ./results    Save analysis to custom directory
  python run.py all                           Run full pipeline end to end
        """
    )

    # Positional arg selects which pipeline to run
    parser.add_argument(
        'pipeline',
        choices=['sam3', 'yolo', 'yolo-sam3', 'analyze', 'all'],
        help="Which pipeline to run"
    )

    # Optional overrides (fall back to config.yaml values if not provided)
    parser.add_argument('--config', default='config.yaml', help="Path to config YAML")
    parser.add_argument('--participants', nargs='+', help="Override participant list (e.g. P1 P2)")
    parser.add_argument('--output', help="Override output directory")
    parser.add_argument('--mode', choices=['sequential', 'simultaneous'], help="SAM 3 tracking mode")
    parser.add_argument('--model', help="YOLO model weights path (e.g. yolo26s.pt)")
    parser.add_argument('--epochs', type=int, help="YOLO training epochs")
    parser.add_argument('--single', action='store_true', help="Run on single video only (quick test)")

    args = parser.parse_args()

    # Load config first 
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Config loaded from: {args.config}")
    print(f"Pipeline: {args.pipeline}")

    # Pick specific pipeline to run
    if args.pipeline == 'sam3':
        run_sam3(config, args)

    elif args.pipeline == 'yolo':
        run_yolo(config, args)

    elif args.pipeline == 'yolo-sam3':
        run_yolo_sam3(config, args)

    elif args.pipeline == 'analyze':
        run_analyze(config, args)

    elif args.pipeline == 'all':

        # Run the full pipeline end-to-end: SAM 3 to YOLO to YOLO-SAM3 to Analysis
        print("\nRunning full pipeline: SAM 3 to YOLO to YOLO-SAM3 to Analysis\n")
        run_sam3(config, args)
        run_yolo(config, args)
        run_yolo_sam3(config, args)
        run_analyze(config, args)

    print("\nDone.")


if __name__ == '__main__':
    main()