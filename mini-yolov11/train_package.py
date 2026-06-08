import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(description="Train a clean YOLOv11n baseline for package detection.")
    parser.add_argument("--data", default="dataset/data.yaml", help="Dataset yaml path.")
    parser.add_argument("--model", default="yolo11n.pt", help="Model weights or yaml path.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--device", default="0", help="CUDA device id or cpu.")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers.")
    parser.add_argument("--project", default="runs/baseline", help="Output project directory.")
    parser.add_argument("--name", default="yolo11n_dataset_baseline", help="Experiment name.")
    return parser.parse_args()


def main():
    args = parse_args()

    from ultralytics.models import YOLO

    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=50,
        save=True,
        save_period=10,
        val=True,
        plots=True,
        verbose=True,
        project=args.project,
        name=args.name,
        pretrained=True,
    )

    save_dir = getattr(results, "save_dir", None)
    print("\nTraining finished.")
    if save_dir:
        print(f"Results: {save_dir}")
        print(f"Best weights: {save_dir}/weights/best.pt")
        print(f"Last weights: {save_dir}/weights/last.pt")


if __name__ == "__main__":
    main()
