import torch
import torch.nn as nn
import timm  # pip install timm
import math
import os

class SwinV2MultiHead(nn.Module):
    def __init__(self, num_classes_a=400, num_classes_b=5000):
        super().__init__()
        # 直接用 SwinV2-Large 作为 backbone
        # 常见版本：'swinv2_large_window12_384'
        self.backbone = timm.create_model(
            'swinv2_large_window12to24_192to384',
            pretrained=True,
            num_classes=0   # 不要默认分类头，只输出特征
        )
        print("timm version:", timm.__version__)
        print("pretrained cfg:", getattr(self.backbone, "pretrained_cfg", None))
        hidden_dim = self.backbone.num_features

        # 两个分类头
        self.head_a = nn.Linear(hidden_dim, num_classes_a)
        self.head_b = nn.Linear(hidden_dim, num_classes_b)

    def forward(self, x, ds_flag=0):
        feats = self.backbone(x)  # 提取特征
        if ds_flag == 0:
            return self.head_a(feats)
        else:
            return self.head_b(feats)

# ================== 调度：warmup + cosine（按step） ==================
def set_cosine_lr(optimizer, step, total_steps, warmup_steps, base_lr, head_lr_mult, lr_floor_ratio=0.05):
    if step < warmup_steps:
        lr_mult = float(step + 1) / float(max(1, warmup_steps))
    else:
        t = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        lr_mult = 0.5 * (1.0 + math.cos(math.pi * t))
    lr_mult = max(lr_mult, lr_floor_ratio)

    # 约定：param_groups[0] = backbone_last_layer, param_groups[1] = heads
    for i, g in enumerate(optimizer.param_groups):
        base = base_lr if i == 0 else base_lr * head_lr_mult
        g["lr"] = base * lr_mult
