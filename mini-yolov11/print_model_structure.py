import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(description="Print YOLO model structure from a YAML config.")
    parser.add_argument(
        "--model",
        default="ultralytics/cfg/models/11/yolo11n-pki.yaml",
        help="Model YAML path.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Image size used for GFLOPs calculation.")
    return parser.parse_args()


def module_args(module):
    """Return a compact description of common YOLO module arguments."""
    args = []
    if hasattr(module, "cv1") and hasattr(module.cv1, "conv"):
        args.append(f"cv1={module.cv1.conv.in_channels}->{module.cv1.conv.out_channels}")
    if hasattr(module, "cv2") and hasattr(module.cv2, "conv"):
        args.append(f"cv2={module.cv2.conv.in_channels}->{module.cv2.conv.out_channels}")
    if hasattr(module, "cv3") and hasattr(module.cv3, "conv"):
        args.append(f"cv3={module.cv3.conv.in_channels}->{module.cv3.conv.out_channels}")
    if hasattr(module, "conv"):
        conv = module.conv
        if hasattr(conv, "in_channels"):
            args.append(f"conv={conv.in_channels}->{conv.out_channels}, k={conv.kernel_size}, s={conv.stride}")
    if hasattr(module, "m"):
        args.append(f"blocks={len(module.m) if hasattr(module.m, '__len__') else type(module.m).__name__}")
    if hasattr(module, "nc"):
        args.append(f"nc={module.nc}")
    return ", ".join(args) if args else "-"


def print_layers(model):
    """Print one row for each top-level YOLO layer."""
    layers = model.model.model
    print("\nModel layers:")
    print(f"{'idx':>4} {'from':>12} {'params':>12} {'module':<45} {'details'}")
    print("-" * 100)
    for layer in layers:
        idx = getattr(layer, "i", "-")
        source = getattr(layer, "f", "-")
        params = getattr(layer, "np", sum(p.numel() for p in layer.parameters()))
        module_type = getattr(layer, "type", layer.__class__.__name__)
        print(f"{idx:>4} {str(source):>12} {params:>12,} {module_type:<45} {module_args(layer)}")


def main():
    args = parse_args()

    from ultralytics.models import YOLO

    model = YOLO(args.model)
    print_layers(model)
    model.info(verbose=True, imgsz=args.imgsz)


if __name__ == "__main__":
    main()
"""
YOLO11n summary: 182 layers, 2,590,425 parameters, 2,590,409 gradients, 6.4 GFLOPs
YOLO11n-pki summary: 192 layers, 2,629,273 parameters, 2,629,257 gradients, 6.6 GFLOPs



"""