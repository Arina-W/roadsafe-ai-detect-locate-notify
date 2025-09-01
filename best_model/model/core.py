from typing import Dict, Union, Optional, Tuple
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights

# ----- Labels (fixed order as trained)
MATERIAL_NAMES = ["asphalt", "concrete", "paving_stones", "unpaved", "sett"]
QUALITY_NAMES  = ["excellent", "good", "intermediate", "bad", "very_bad"]

# ----- Net (same heads as training)
class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        base = efficientnet_b7(weights=EfficientNet_B7_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(base.children())[:-1])
        in_feats = base.classifier[1].in_features  # typically 2560
        self.mat = nn.Linear(in_feats, len(MATERIAL_NAMES))
        self.qual = nn.Linear(in_feats, len(QUALITY_NAMES))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.features(x).flatten(1)
        return self.mat(x), self.qual(x)

# ----- Preprocess (must match training)
_pre = transforms.Compose([
    transforms.Resize((600, 600)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def _load_clean_state_dict(weights_path: str, map_location) -> dict:
    """
    Loads a checkpoint and normalizes key names so it matches an uncompiled, non-DataParallel model:
    - Strips '_orig_mod.' (from torch.compile)
    - Strips 'module.' (from nn.DataParallel / DDP)
    - Unwraps {'state_dict': ...} if present
    """
    sd = torch.load(weights_path, map_location=map_location)

    # Unwrap common containers
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]

    if not isinstance(sd, dict):
        raise RuntimeError(f"Unexpected checkpoint format at {weights_path}")

    cleaned = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("_orig_mod."):
            nk = nk[len("_orig_mod."):]
        if nk.startswith("module."):
            nk = nk[len("module."):]
        cleaned[nk] = v
    return cleaned

# ----- Loader + one-shot API
class Session:
    def __init__(self, weights: str, device: Optional[str] = None) -> None:
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(dev)
        self.model = Net().to(self.device)

        # Clean & load checkpoint to handle torch.compile/DataParallel saves
        state = _load_clean_state_dict(weights, map_location=self.device)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            # Non-fatal; print once so users know if anything looked odd.
            print("⚠️ load_state_dict warnings:")
            if missing:   print("  missing:", missing)
            if unexpected: print("  unexpected:", unexpected)

        self.model.eval()

    @torch.inference_mode()
    def __call__(self, image: Union[str, Image.Image]) -> Dict[str, object]:
        img = Image.open(image).convert("RGB") if isinstance(image, str) else image.convert("RGB")
        x = _pre(img).unsqueeze(0).to(self.device)
        m, q = self.model(x)
        m_p = torch.softmax(m, -1)[0]
        q_p = torch.softmax(q, -1)[0]
        mi = int(m_p.argmax())
        qi = int(q_p.argmax())

        result = {
            "surface_type_idx": mi,
            "surface_type": MATERIAL_NAMES[mi],
            "surface_type_probs": [float(p) for p in m_p.cpu().tolist()],
            "surface_quality_idx": qi,
            "surface_quality": QUALITY_NAMES[qi],
            "surface_quality_probs": [float(p) for p in q_p.cpu().tolist()],
        }

        # 🔎 Print full result to console for debugging/demo
        print("🔎 RoadSafe prediction result:")
        for k, v in result.items():
            print(f"  {k}: {v}")

        return result

_DEF_W = "weights/best_model.pt"

def load(weights: str = _DEF_W, device: Optional[str] = None) -> Session:
    return Session(weights, device)

@torch.inference_mode()
def predict(image: Union[str, Image.Image], weights: str = _DEF_W, device: Optional[str] = None):
    return Session(weights, device)(image)
