"""PyTorch dataset, transforms and device helpers for deep models."""
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from src import config as C


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class CarsDataset(Dataset):
    def __init__(self, meta: pd.DataFrame, class_to_idx: dict[str, int], transform):
        self.paths = meta["path"].tolist()
        self.labels = meta["make"].map(class_to_idx).tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(C.IMAGES_DIR / self.paths[i]).convert("RGB")
        return self.transform(img), self.labels[i]


def build_transforms(size: int, mean, std):
    # Photos are ~1.5:1; crops keep a car-like aspect ratio and are then resized to a square input.
    train = T.Compose([
        T.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(1.2, 1.8)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.2, 0.2, 0.2, 0.02),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    evaluation = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    return train, evaluation


def build_loaders(meta: pd.DataFrame, class_names: list[str], size: int, mean, std,
                  batch_size: int, workers: int, limit: int | None = None):
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    train_tf, eval_tf = build_transforms(size, mean, std)
    loaders = {}
    for split in ("train", "val", "test"):
        part = meta[meta["split"] == split]
        if limit:
            part = part.sample(min(limit, len(part)), random_state=C.SEED)
        ds = CarsDataset(part, class_to_idx, train_tf if split == "train" else eval_tf)
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=split == "train",
                                    num_workers=workers, pin_memory=torch.cuda.is_available(),
                                    persistent_workers=workers > 0, drop_last=split == "train")
    return loaders
