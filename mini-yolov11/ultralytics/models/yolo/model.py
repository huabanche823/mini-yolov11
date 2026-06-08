# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import ROOT, YAML


class YOLO(Model):
    """YOLO (You Only Look Once) object detection model.

    This class provides a unified interface for YOLO object detection models.

    Attributes:
        model: The loaded YOLO model instance.
        task: The task type (detect).
        overrides: Configuration overrides for the model.

    Methods:
        __init__: Initialize a YOLO model.
        task_map: Map tasks to their corresponding model, trainer, validator, and predictor classes.

    Examples:
        Load a YOLOv11 detection model from config
        >>> model = YOLO("ultralytics/cfg/models/11/yolo11.yaml")

        Predict on an image
        >>> model = YOLO("yolo11n.pt")
        >>> results = model.predict("image.jpg")
    """

    def __init__(self, model: str | Path = "yolo11n.pt", task: str | None = None, verbose: bool = False):
        """Initialize a YOLO model.

        Args:
            model (str | Path): Model name or path to model file, i.e. 'yolo11n.pt', 'yolo11n.yaml'.
            task (str, optional): YOLO task specification, defaults to 'detect'.
            verbose (bool): Display model info on load.
        """
        # Initialize with default YOLO behavior (detect task only)
        super().__init__(model=model, task=task or "detect", verbose=verbose)

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "detect": {
                "model": DetectionModel,
                "trainer": yolo.detect.DetectionTrainer,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            }
        }
