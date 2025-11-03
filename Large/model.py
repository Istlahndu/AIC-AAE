import torch
import torch.nn as nn
import timm  
import math
import os

class SwinV2MultiHead(nn.Module):
    def __init__(self, num_classes_a=400, num_classes_b=5000):
        super().__init__()
        self.backbone = timm.create_model(
            'swinv2_large_window12to24_192to384',
            pretrained=False,
            num_classes=0   
        )
        print("timm version:", timm.__version__)
        print("pretrained cfg:", getattr(self.backbone, "pretrained_cfg", None))
        hidden_dim = self.backbone.num_features

        self.head_a = nn.Linear(hidden_dim, num_classes_a)
        self.head_b = nn.Linear(hidden_dim, num_classes_b)

    def forward(self, x, ds_flag=0):
        feats = self.backbone(x)  
        if ds_flag == 0:
            return self.head_a(feats)
        else:
            return self.head_b(feats)

def set_cosine_lr(optimizer, step, total_steps, warmup_steps, base_lr, head_lr_mult, lr_floor_ratio=0.05):
    if step < warmup_steps:
        lr_mult = float(step + 1) / float(max(1, warmup_steps))
    else:
        t = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        lr_mult = 0.5 * (1.0 + math.cos(math.pi * t))
    lr_mult = max(lr_mult, lr_floor_ratio)

    for i, g in enumerate(optimizer.param_groups):
        base = base_lr if i == 0 else base_lr * head_lr_mult
        g["lr"] = base * lr_mult
