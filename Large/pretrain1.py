# train_B.py (Large + Gradient Accumulation + 新AMP API)
import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast          # 新API
from timm.utils import ModelEmaV2
from PIL import ImageFile

from datapre import DataPreparer
from model import SwinV2MultiHead

import sys, atexit
from datetime import datetime

LOG_TXT_PATH = '../logs/pretrain5W_print_2.txt'
os.makedirs(os.path.dirname(LOG_TXT_PATH), exist_ok=True)

class Tee:
    def __init__(self, f):
        self.f = f
        self._stdout = sys.__stdout__
    def write(self, x):
        self._stdout.write(x)
        self.f.write(x)
    def flush(self):
        self._stdout.flush()
        self.f.flush()

_log_f = open(LOG_TXT_PATH, 'a', buffering=1)
_log_f.write(f"\n===== New run: {datetime.now().isoformat()} =====\n")
sys.stdout = Tee(_log_f)
sys.stderr = sys.stdout
atexit.register(_log_f.close)

def unwrap(m):
    return m.module if isinstance(m, nn.DataParallel) else m

# ================== 基础配置 ==================
DATASET        = '../data/train_B'
BATCH_SIZE     = 12                      # 每步喂入的micro-batch
ACCUM_STEPS    = 3                       # 梯度累计步数 → 有效batch=12*3=36
TOTAL_ITERS    = 40000                   # 这里指“优化器 step 的总次数”（全局步）
BASE_LR        = 2e-6                    # backbone LR（固定值更稳）
HEAD_LR_MULT   = 5.0                     # 头部LR倍率
WEIGHT_DECAY   = 5e-2
WARMUP_ITERS   = 500                     # 按“优化器 step”计数
EMA_DECAY      = 0.9995
MAX_NORM       = 1.0
CKPT_PATH      = '../model2/Large_1.pth'
SAVE_PATH      = '../model2/Large_2.pth'
EVAL_INTERVAL  = 200                     # 按“优化器 step”间隔评估
PRINT_INTERVAL = 50                      # 按“优化器 step”打印
NUM_SAMPLED_VAL= 1000
SAVE_INTERVAL  = 800

# === 新增：最优权重保存路径（仅保存，不在恢复时加载/使用） ===
BEST_SAVE_PATH = '../model2/Large_best2.pth'

torch.set_float32_matmul_precision('high')

# ================== 调度：warmup + cosine（按step） ==================
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

# ================== 评估（默认用EMA更稳） ==================
@torch.no_grad()
def evaluate(model_for_eval, loader, device='cuda', ds_flag=1):
    model_for_eval.eval()
    correct = total = 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast('cuda'):
            logits = model_for_eval(imgs, ds_flag=ds_flag)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total   += labels.size(0)
    return correct / max(1, total)

def train():
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ========== 数据（384，匹配 Large） ==========
    data_prep    = DataPreparer(DATASET, batch_size=BATCH_SIZE)   # img_size 在 datapre.py 中固定 384
    train_loader = data_prep.get_train_loader()

    # ========== 模型 ==========
    base_model = SwinV2MultiHead(
        num_classes_a=400,
        num_classes_b=data_prep.get_num_classes()
    ).to(device)

    if torch.cuda.device_count() > 1:
        print(f"[DP] Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(base_model)
    else:
        model = base_model

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 分组LR：backbone / heads
    backbone_params = (model.module.backbone if isinstance(model, nn.DataParallel) else model.backbone).parameters()
    head_a_params   = (model.module.head_a if isinstance(model, nn.DataParallel) else model.head_a).parameters()
    head_b_params   = (model.module.head_b if isinstance(model, nn.DataParallel) else model.head_b).parameters()

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": BASE_LR, "weight_decay": WEIGHT_DECAY},
            {"params": list(head_a_params) + list(head_b_params),
             "lr": BASE_LR * HEAD_LR_MULT, "weight_decay": WEIGHT_DECAY},
        ],
        betas=(0.9, 0.999)
    )

    scaler = GradScaler('cuda')
    ema = ModelEmaV2(base_model, decay=EMA_DECAY)

    # ========== 断点恢复 ==========
    # 说明：我们将 TOTAL_ITERS 视为“优化器step计数”；因此恢复 global_step。
    global_step = 0
    micro_step  = 0
    if os.path.exists(CKPT_PATH):
        ckpt = torch.load(CKPT_PATH, map_location='cpu')
        base_model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scaler.load_state_dict(ckpt['scaler'])
        if 'ema' in ckpt and ckpt['ema'] is not None:
            ema.load_state_dict(ckpt['ema'])
        global_step = ckpt.get('iter', 0)   # 上次已经完成的“优化器step”数量
        print(f"[Resume] resume from global_step={global_step}")

    print("Starting training on train_B (SwinV2-Large@384) with Gradient Accumulation...")
    model.train()

    train_iter = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)

    # === 新增：在本次运行内追踪最佳精度（不参与恢复，只用于保存） ===
    best_acc = -1.0
    best_iter = -1

    while global_step < TOTAL_ITERS:
        # ===== 取一个 micro-batch =====
        try:
            imgs, labels = next(train_iter)
        except StopIteration:
            train_loader = data_prep.get_train_loader()
            train_iter = iter(train_loader)
            imgs, labels = next(train_iter)

        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # ===== 前向 + 计算 loss （记得 /ACCUM_STEPS）=====
        with autocast('cuda'):
            logits = model(imgs, ds_flag=1)   # 只训练 B 头
            loss   = criterion(logits, labels) / ACCUM_STEPS

        # ===== 反向（累计梯度，不立刻 step）=====
        scaler.scale(loss).backward()
        micro_step += 1

        # ===== 累计到 ACCUM_STEPS 次，再做一次“有效 step” =====
        if micro_step % ACCUM_STEPS == 0:
            # 先unscale再裁剪
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # EMA、LR调度都在真正 step 后进行
            ema.update(unwrap(model))

            set_cosine_lr(
                optimizer,
                step=global_step,              # 第 global_step 次 step 已完成
                total_steps=TOTAL_ITERS,
                warmup_steps=WARMUP_ITERS,
                base_lr=BASE_LR,
                head_lr_mult=HEAD_LR_MULT
            )

            global_step += 1

            # ===== 打印 =====
            if global_step % PRINT_INTERVAL == 0 or global_step == TOTAL_ITERS:
                lr_back = optimizer.param_groups[0]["lr"]
                lr_head = optimizer.param_groups[1]["lr"]
                print(f"Iter {global_step}/{TOTAL_ITERS} | "
                      f"Loss {loss.item() * ACCUM_STEPS:.4f} | "  # 还原为每micro-batch未除的loss观感
                      f"LR_b {lr_back:.2e} | LR_h {lr_head:.2e}")

            # ===== 验证（EMA）=====
            if global_step % EVAL_INTERVAL == 0 or global_step == TOTAL_ITERS:
                test_loader = data_prep.get_sampled_test_loader(num_samples=NUM_SAMPLED_VAL)
                acc = evaluate(ema.module, test_loader, device=device, ds_flag=1)
                print(f"[Eval-EMA] Iter {global_step} Acc: {acc:.6f}")  # 多几位小数避免 0.0000 错觉

                # === 新增：如果精度更好，则额外保存最优权重（仅保存，不恢复使用） ===
                if acc > best_acc:
                    old_best = best_acc
                    best_acc = acc
                    best_iter = global_step
                    os.makedirs(os.path.dirname(BEST_SAVE_PATH), exist_ok=True)
                    torch.save({
                        'iter': global_step,
                        'model': unwrap(model).state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scaler': scaler.state_dict(),
                        'ema': ema.state_dict(),
                        # 仅作为信息记录，后续不用于恢复
                        'best_acc': best_acc,
                        'best_iter': best_iter,
                    }, BEST_SAVE_PATH)
                    print(f"[BEST] Acc improved {old_best:.6f} → {best_acc:.6f} at iter {best_iter}. "
                          f"Saved BEST to {BEST_SAVE_PATH}")

            # ===== 保存 =====
            if global_step % SAVE_INTERVAL == 0 or global_step == TOTAL_ITERS:
                os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
                torch.save({
                    'iter': global_step,                          # 记录“优化器step”
                    'model': unwrap(model).state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                    'ema': ema.state_dict(),
                }, SAVE_PATH)
                print(f"[Save] Saved to {SAVE_PATH}")

if __name__ == '__main__':
    train()
