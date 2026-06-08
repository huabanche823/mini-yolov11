# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# 只导入可用的模型（简化版）
from .yolo import YOLO

__all__ = "YOLO"  # allow simpler import
