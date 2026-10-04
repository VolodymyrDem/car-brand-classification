"""Load trained deep models and predict the car make for arbitrary photos."""
import timm
import torch
from PIL import Image

from src import config as C
from src.data import build_transforms, get_device

MODELS = {
    "resnet18": ("ResNet18", "resnet18.a1_in1k"),
    "vit_small": ("ViT-Small", "vit_small_patch16_224.augreg_in21k_ft_in1k"),
}


def class_names() -> list[str]:
    # Same order as during training: sorted makes (make_counts_by_split.csv is tracked in git).
    import pandas as pd
    return sorted(pd.read_csv(C.TABLES_DIR / "make_counts_by_split.csv")["make"])


class Predictor:
    def __init__(self, device=None):
        self.device = device or get_device()
        self.classes = class_names()
        self.models, self.transforms = {}, {}
        for key, (_, timm_name) in MODELS.items():
            path = C.MODELS_DIR / f"{key}.pt"
            if not path.exists():
                continue
            model = timm.create_model(timm_name, pretrained=False, num_classes=len(self.classes))
            model.load_state_dict(torch.load(path, map_location="cpu"))
            self.models[key] = model.eval().to(self.device)
            cfg = timm.data.resolve_model_data_config(model)
            self.transforms[key] = build_transforms(cfg["input_size"][-1], cfg["mean"], cfg["std"])[1]
        if not self.models:
            raise FileNotFoundError(f"No models/*.pt found in {C.MODELS_DIR}")

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict[str, dict[str, float]]:
        """Return {model_name: {make: probability}} plus an 'Ensemble' entry when both models are loaded."""
        image = image.convert("RGB")
        out = {}
        for key, model in self.models.items():
            x = self.transforms[key](image).unsqueeze(0).to(self.device)
            out[key] = model(x).softmax(1)[0].cpu()
        result = {MODELS[k][0]: dict(zip(self.classes, p.tolist())) for k, p in out.items()}
        if len(out) > 1:
            ens = torch.stack(list(out.values())).mean(0)
            result["Ensemble"] = dict(zip(self.classes, ens.tolist()))
        return result
