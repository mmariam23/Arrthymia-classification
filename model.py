"""
pipeline/model.py
─────────────────
VGG-16 1-D arrhythmia classifier.
Architecture copied verbatim from the training notebook (cfg['D']).

Input  : (N, 1, 1000)  float32   min-max normalised, 100 Hz, 10 s
Classes: SR | PVC | PAC | VT | SVT | AF   (indices 0–5)
"""

import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger      = logging.getLogger(__name__)
CLASS_NAMES = ["SR", "PVC", "PAC", "VT", "SVT", "AF"]
SEG_LEN     = 1000


# ── Architecture (matches notebook exactly) ───────────────────────

def _make_layers(cfg_list: list, batch_norm: bool = True) -> nn.Sequential:
    layers, in_ch = [], 1
    for v in cfg_list:
        if v == "M":
            layers.append(nn.MaxPool1d(kernel_size=3, stride=3))
        else:
            conv = nn.Conv1d(in_ch, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv, nn.BatchNorm1d(v), nn.ReLU(inplace=True)]
                if v >= 128:
                    layers.append(nn.Dropout1d(p=0.1))
            else:
                layers += [conv, nn.ReLU(inplace=True)]
            in_ch = v
    return nn.Sequential(*layers)


_CFG_D = [32, 32, "M", 64, 64, "M", 128, 128, 128, "M",
          256, 256, 256, "M", 256, 256, 256, "M"]


class _VGG(nn.Module):
    def __init__(self, features: nn.Sequential, num_classes: int = 6):
        super().__init__()
        self.features   = features
        self.classifier = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ── Wrapper ───────────────────────────────────────────────────────

class VGGModel:
    def __init__(self, model_path: str, num_classes: int = 6):
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.is_loaded   = False
        self._net        = None
        self._load(model_path)

    def _load(self, path: str) -> None:
        if not os.path.exists(path):
            logger.warning(f"No model found at '{path}'. Place vgg_arrhythmia.pt in model/ folder.")
            return
        try:
            logger.info(f"Loading model from '{path}' ...")
            state = torch.load(path, map_location=self.device, weights_only=False)
            logger.info(f"Checkpoint type: {type(state)}")

            net = _VGG(_make_layers(_CFG_D), num_classes=self.num_classes)

            if isinstance(state, dict) and "state_dict" in state:
                logger.info("Detected wrapped checkpoint with 'state_dict' key")
                net.load_state_dict(state["state_dict"])

            elif isinstance(state, dict):
                logger.info(f"Detected plain state_dict with {len(state)} keys")
                # Strip 'module.' prefix if saved with DataParallel
                cleaned = {k.replace("module.", ""): v for k, v in state.items()}
                net.load_state_dict(cleaned)

            elif isinstance(state, nn.Module):
                logger.info("Detected full model object")
                net = state

            else:
                raise ValueError(f"Unrecognised checkpoint format: {type(state)}")

            net.to(self.device).eval()
            self._net      = net
            self.is_loaded = True
            logger.info(f"VGG model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load model from '{path}': {e}", exc_info=True)

    def predict_single(self, seg_norm: np.ndarray) -> tuple[str, float]:
        if not self.is_loaded:
            return "Model unavailable", 0.0
        try:
            x     = self._to_tensor(seg_norm)
            with torch.no_grad():
                probs = F.softmax(self._net(x), dim=1).cpu().numpy()[0]
            idx   = int(np.argmax(probs))
            label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"Class {idx}"
            return label, float(probs[idx])
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return "Prediction error", 0.0

    def _to_tensor(self, seg: np.ndarray) -> torch.Tensor:
        seg = np.asarray(seg, dtype=np.float32)
        if len(seg) >= SEG_LEN:
            seg = seg[:SEG_LEN]
        else:
            seg = np.pad(seg, (0, SEG_LEN - len(seg)))
        return torch.from_numpy(seg.reshape(1, 1, SEG_LEN)).to(self.device)