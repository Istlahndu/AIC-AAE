# infer_multihead_nocli.py
import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import timm
from timm.data import resolve_model_data_config, create_transform


class SwinV2MultiHead(nn.Module):
    def __init__(self, num_classes_a=400, num_classes_b=5000):
        super().__init__()
        self.backbone = timm.create_model(
            'swinv2_large_window12to24_192to384',
            pretrained=True,
            num_classes=0
        )
        hidden_dim = self.backbone.num_features
        self.head_a = nn.Linear(hidden_dim, num_classes_a)
        self.head_b = nn.Linear(hidden_dim, num_classes_b)

    def forward(self, x, ds_flag=0):
        feats = self.backbone(x)
        if ds_flag == 0:
            return self.head_a(feats)
        else:
            return self.head_b(feats)


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


class ImageFolderFlat(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = sorted([p for p in self.root.rglob("*") if p.suffix.lower() in IMG_EXTS])
        if len(self.samples) == 0:
            raise FileNotFoundError(f"No images found in {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, path.name


def load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "ema" in ckpt and isinstance(ckpt["ema"], dict) and "module" in ckpt["ema"]:
        model.load_state_dict(ckpt["ema"]["module"], strict=False)
        print(f"[INFO] Loaded EMA weights from {ckpt_path}")
    elif isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=False)
        print(f"[INFO] Loaded MODEL weights from {ckpt_path}")
    else:
        model.load_state_dict(ckpt, strict=False)
        print(f"[INFO] Loaded plain state_dict from {ckpt_path}")


@torch.inference_mode()
def run_infer(model, dataloader, device, ds_flag):
    model.eval()
    results = []
    for imgs, names in dataloader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float16):
            logits = model(imgs, ds_flag=ds_flag)
            preds = logits.argmax(dim=1).tolist()
        for fname, p in zip(names, preds):
            results.append((fname, f"{p:04d}"))
    results.sort(key=lambda x: x[0])
    return results


def write_csv(results, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for fname, cls4 in results:
            f.write(f"{fname}, {cls4}\n")
    print(f"[OK] {out_path} written, {len(results)} lines")


def main():
    # test_5000_dir = "../data/test_B"  # 初赛5000
    # test_400_dir = "../data/test_A"  # 初赛400
    test_5000_dir = "../data/data2/test_5000"  # 复赛5000
    test_400_dir = "../data/data2/test_400"  # 复赛400
    ckpt_5000 = "../model2/Large_best2.pth"
    ckpt_400 = "../model2/Large_best0.pth"
    out_5000 = "preds_5000.csv"
    out_400 = "preds_400.csv"
    batch_size = 32
    num_workers = 4

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    model = SwinV2MultiHead(num_classes_a=400, num_classes_b=5000).to(device)
    transform = create_transform(**resolve_model_data_config(model.backbone))

    # ---------- 5000 ----------
    print("\n[Stage] 5000-class inference")
    load_checkpoint(model, ckpt_5000, device)
    ds_5000 = ImageFolderFlat(test_5000_dir, transform)
    dl_5000 = DataLoader(ds_5000, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    results_5000 = run_infer(model, dl_5000, device, ds_flag=1)
    write_csv(results_5000, out_5000)

    # ---------- 400 ----------
    print("\n[Stage] 400-class inference")
    load_checkpoint(model, ckpt_400, device)
    ds_400 = ImageFolderFlat(test_400_dir, transform)
    dl_400 = DataLoader(ds_400, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    results_400 = run_infer(model, dl_400, device, ds_flag=0)
    write_csv(results_400, out_400)

    print("\n[Done] All predictions saved.")


if __name__ == "__main__":
    main()
