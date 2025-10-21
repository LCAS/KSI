# models/perceptree.py
import os
import sys
import cv2
import torch
import numpy as np
import subprocess
from pathlib import Path
from typing import List, Optional

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

MODEL_NAME = "ResNext-101_fold_01.pth"  # default path: ./output/ResNext-101_fold_01.pth
GDRIVE_FILE_ID = "108tORWyD2BFFfO5kYim9jP0wIVNcw0OJ"  # from your link

# --- Minimal YOLO-like result wrappers ----------------------------------------
class _Masks:
    def __init__(self, data_np: np.ndarray):
        if data_np.dtype != np.uint8:
            data_np = data_np.astype(np.uint8)
        data_np = (data_np > 0).astype(np.uint8)   # ensure {0,1}
        self.data = torch.from_numpy(data_np)      # torch.uint8 [N,H,W]


class _Result:
    def __init__(self, masks_np: Optional[np.ndarray],
                 scores_np: Optional[np.ndarray] = None,
                 classes_np: Optional[np.ndarray] = None):
        self.masks = _Masks(masks_np) if (masks_np is not None and masks_np.size > 0) else None
        self.scores = torch.from_numpy(scores_np) if scores_np is not None else None
        self.cls = torch.from_numpy(classes_np) if classes_np is not None else None


def _ensure_gdown_available():
    """Import gdown, installing it if necessary."""
    try:
        import gdown  # noqa: F401
        return
    except ImportError:
        print("[PercepTree] 'gdown' not found. Installing...", file=sys.stderr)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown  # noqa: F401
        return


def _download_from_gdrive(file_id: str, output_path: Path):
    """Download a file from Google Drive to output_path using gdown."""
    _ensure_gdown_available()
    import gdown
    output_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"[PercepTree] Downloading weights from Google Drive -> {output_path}")
    gdown.download(url, str(output_path), quiet=False)  # handles confirmation for large files


# --- PercepTree wrapper that mimics Ultralytics YOLO --------------------------
class PercepTree:
    def __init__(
        self,
        weights_path: Optional[str] = None,
        num_classes: int = 1,
        score_thresh: float = 0.7,
        device: Optional[str] = None,
    ):
        """
        Detectron2 wrapper that emulates YOLO's predict() return for masks.

        Args:
          weights_path: path to .pth (defaults to ./output/MODEL_NAME)
          num_classes: number of classes Detectron2 head predicts (default 1)
          score_thresh: ROI_HEADS.SCORE_THRESH_TEST
          device: 'cuda', 'cpu', or None -> auto
        """
        # Resolve weights path and ensure the file exists (download if missing)
        if weights_path is None:
            weights_path = str(Path("./output") / MODEL_NAME)
        weights_path = str(Path(weights_path))  # normalize

        weights_file = Path(weights_path)
        if not weights_file.exists():
            try:
                _download_from_gdrive(GDRIVE_FILE_ID, weights_file)
            except Exception as e:
                raise FileNotFoundError(
                    f"[PercepTree] Could not find or download weights to '{weights_path}'. "
                    f"Original error: {e}"
                ) from e
            if not weights_file.exists():
                raise FileNotFoundError(
                    f"[PercepTree] Download reported success but weights still not found at '{weights_path}'."
                )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        cfg = get_cfg()
        cfg.INPUT.MASK_FORMAT = "bitmask"
        cfg.merge_from_file(
            model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml")
        )
        cfg.DATASETS.TRAIN = ()
        cfg.DATASETS.TEST = ()
        cfg.DATALOADER.NUM_WORKERS = 4
        cfg.MODEL.MASK_ON = True
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(score_thresh)
        cfg.MODEL.WEIGHTS = weights_path
        cfg.MODEL.DEVICE = device

        self.cfg = cfg
        print(f"[PercepTree] Device={self.cfg.MODEL.DEVICE}  Weights={weights_path}")
        self.predictor = DefaultPredictor(self.cfg)

    def predict(
        self,
        img_bgr: np.ndarray,
        conf: Optional[float] = None,
        classes: Optional[List[int]] = None,
        verbose: bool = False,
    ):
        if conf is not None and conf != self.predictor.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST:
            self.predictor.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(conf)

        outputs = self.predictor(img_bgr)
        inst = outputs["instances"].to("cpu")

        if len(inst) == 0 or not inst.has("pred_masks"):
            return [_Result(None)]

        masks = inst.pred_masks.numpy().astype(np.uint8)  # [N,H,W] in {0,1}
        scores = inst.scores.numpy() if inst.has("scores") else None
        cls = inst.pred_classes.numpy() if inst.has("pred_classes") else None

        if classes is not None and cls is not None and masks is not None:
            keep = np.isin(cls, np.array(classes, dtype=cls.dtype))
            masks = masks[keep]
            if scores is not None:
                scores = scores[keep]
            cls = cls[keep]

        return [_Result(masks, scores, cls)]

    def to(self, device: str):
        if device and device != self.cfg.MODEL.DEVICE:
            print(f"[PercepTree] Warning: .to('{device}') is a no-op; device was set at init to '{self.cfg.MODEL.DEVICE}'.")
        return self
