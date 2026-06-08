"""
YOLOv11 Inference Test Script

This script tests the YOLOv11 model inference functionality using the mini-yolov11 codebase.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from pathlib import Path
from PIL import Image

def test_model_creation():
    """Test that the YOLOv11 model can be created from YAML config."""
    print("\n" + "="*60)
    print("Test 1: Model Creation from YAML")
    print("="*60)

    try:
        from ultralytics.nn.tasks import DetectionModel

        model_cfg = "ultralytics/cfg/models/11/yolo11.yaml"
        model = DetectionModel(cfg=model_cfg, nc=80, verbose=True)

        print("\n[PASS] Model created successfully!")
        print(f"  Model type: {type(model)}")
        print(f"  Number of classes: {model.yaml['nc']}")
        print(f"  Stride: {model.stride.tolist()}")

        return model
    except Exception as e:
        print(f"\n[FAIL] Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_forward_pass(model):
    """Test that the model can perform a forward pass."""
    print("\n" + "="*60)
    print("Test 2: Forward Pass")
    print("="*60)

    if model is None:
        print("[FAIL] Skipped - no model available")
        return None

    try:
        model.eval()

        dummy_input = torch.randn(1, 3, 640, 640)

        print(f"\n  Input shape: {dummy_input.shape}")

        with torch.no_grad():
            output = model(dummy_input)

        print(f"\n  Output type: {type(output)}")

        if isinstance(output, torch.Tensor):
            print(f"  Output shape: {output.shape}")
            print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")
        elif isinstance(output, tuple):
            print(f"  Number of outputs: {len(output)}")
            for i, o in enumerate(output):
                if isinstance(o, torch.Tensor):
                    print(f"    Output {i} shape: {o.shape}")
                elif isinstance(o, dict):
                    print(f"    Output {i} keys: {list(o.keys())}")
                    for k, v in o.items():
                        if isinstance(v, torch.Tensor):
                            print(f"      {k} shape: {v.shape}")
        elif isinstance(output, dict):
            print(f"  Output keys: {list(output.keys())}")
            for k, v in output.items():
                if isinstance(v, torch.Tensor):
                    print(f"    {k} shape: {v.shape}")

        print("\n[PASS] Forward pass successful!")
        return output

    except Exception as e:
        print(f"\n[FAIL] Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_image_preprocessing():
    """Test image loading and preprocessing."""
    print("\n" + "="*60)
    print("Test 3: Image Preprocessing")
    print("="*60)

    try:
        image_path = Path(__file__).parent.parent / "ultralytics" / "assets" / "bus.jpg"

        if not image_path.exists():
            print(f"[FAIL] Test image not found: {image_path}")
            return None

        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)

        print(f"\n  Original image shape: {img_array.shape}")
        print(f"  Image size: {img.size}")
        print(f"  Image mode: {img.mode}")

        target_size = 640
        from ultralytics.data.augment import LetterBox

        letterbox = LetterBox(new_shape=(target_size, target_size), auto=False, stride=32)
        img_resized = letterbox(image=img_array)

        print(f"  Resized image shape: {img_resized.shape}")

        img_normalized = img_resized.transpose(2, 0, 1)[np.newaxis, ...] / 255.0
        img_tensor = torch.from_numpy(img_normalized).float()

        print(f"  Normalized tensor shape: {img_tensor.shape}")

        print("\n[PASS] Image preprocessing successful!")
        return img_tensor

    except Exception as e:
        print(f"\n[FAIL] Image preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_inference_on_image(model):
    """Test inference on a real image."""
    print("\n" + "="*60)
    print("Test 4: Inference on Real Image")
    print("="*60)

    if model is None:
        print("[FAIL] Skipped - no model available")
        return None

    try:
        from ultralytics.data.augment import LetterBox
        import cv2

        image_path = Path(__file__).parent.parent / "ultralytics" / "assets" / "bus.jpg"
        if not image_path.exists():
            print(f"[FAIL] Test image not found: {image_path}")
            return None

        img0 = cv2.imread(str(image_path))
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)

        print(f"\n  Original image shape: {img0.shape}")

        letterbox = LetterBox(new_shape=(640, 640), auto=False, stride=32)
        img = letterbox(image=img0)

        print(f"  Letterboxed image shape: {img.shape}")

        img = img.transpose(2, 0, 1)[np.newaxis, ...] / 255.0
        img_tensor = torch.from_numpy(img).float()

        print(f"  Input tensor shape: {img_tensor.shape}")

        model.eval()
        with torch.no_grad():
            output = model(img_tensor)

        print(f"\n  Inference completed!")
        print(f"  Output type: {type(output)}")

        if isinstance(output, tuple):
            print(f"  Number of outputs: {len(output)}")
            for i, o in enumerate(output):
                if isinstance(o, torch.Tensor):
                    print(f"    Output {i} shape: {o.shape}, range: [{o.min():.4f}, {o.max():.4f}]")
        elif isinstance(output, torch.Tensor):
            print(f"  Output shape: {output.shape}")
            print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")

        print("\n[PASS] Inference on real image successful!")
        return output

    except Exception as e:
        print(f"\n[FAIL] Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_model_info(model):
    """Test model information."""
    print("\n" + "="*60)
    print("Test 5: Model Information")
    print("="*60)

    if model is None:
        print("[FAIL] Skipped - no model available")
        return

    try:
        print("\n  Model architecture:")
        print(f"    Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"    Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        print(f"\n  Model structure:")
        for i, module in enumerate(model.model):
            if hasattr(module, 'np'):
                print(f"    Layer {i}: {module.type}, params: {module.np:,}")

        print("\n[PASS] Model info retrieved successfully!")

    except Exception as e:
        print(f"\n[FAIL] Failed to get model info: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main test function."""
    print("\n" + "="*60)
    print("YOLOv11 Inference Test Suite")
    print("="*60)

    test_results = {}

    model = test_model_creation()
    test_results['model_creation'] = model is not None

    output = test_forward_pass(model)
    test_results['forward_pass'] = output is not None

    tensor = test_image_preprocessing()
    test_results['image_preprocessing'] = tensor is not None

    output = test_inference_on_image(model)
    test_results['inference_on_image'] = output is not None

    test_model_info(model)

    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    passed = sum(1 for v in test_results.values() if v)
    total = len(test_results)
    print(f"\n  Passed: {passed}/{total}")
    for test_name, result in test_results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"    {status} {test_name}")

    if passed == total:
        print("\n[SUCCESS] All tests passed! YOLOv11 inference is working correctly.")
        return 0
    else:
        print(f"\n[WARNING] {total - passed} test(s) failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
