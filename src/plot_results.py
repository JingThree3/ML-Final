"""
plot_results.py

绘制五个随机种子的预测曲线对比图
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

from dataset import build_dataloader
from models import LSTMModel


# ===========================
# 参数
# ===========================

TASK = "90"

SEEDS = [42, 52, 62, 72, 82]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_SIZE = 13
HIDDEN_SIZE = 64
NUM_LAYERS = 2
OUTPUT_SIZE = 90 if TASK == "90" else 365


MODEL_DIR = "outputs/models"
SAVE_DIR = "outputs/figures"

os.makedirs(SAVE_DIR, exist_ok=True)


# ===========================
# DataLoader
# ===========================

_, test_loader = build_dataloader(
    task=TASK,
    batch_size=32
)


# ===========================
# 取得测试集第一个样本
# ===========================

x_sample = None
y_true = None

for x, y in test_loader:
    x_sample = x[0:1].to(DEVICE)
    y_true = y[0].numpy()
    break


# ===========================
# 开始画图
# ===========================

plt.figure(figsize=(12, 6))

# Ground Truth只画一次
plt.plot(
    y_true,
    color="black",
    linewidth=3,
    label="Ground Truth"
)

line_styles = [
    "--",
    "-.",
    ":",
    (0, (3, 1, 1, 1)),
    (0, (5, 2))
]

colors = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple"
]

for seed, ls, color in zip(SEEDS, line_styles, colors):

    model = LSTMModel(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=OUTPUT_SIZE
    ).to(DEVICE)

    model_path = os.path.join(
        MODEL_DIR,
        f"best_lstm_{TASK}_seed{seed}.pth"
    )

    model.load_state_dict(
        torch.load(model_path, map_location=DEVICE)
    )

    model.eval()

    with torch.no_grad():

        pred = model(x_sample)

    pred = pred.cpu().numpy().flatten()

    plt.plot(
        pred,
        linestyle=ls,
        linewidth=2,
        color=color,
        label=f"Seed {seed}"
    )


plt.title(
    f"LSTM Prediction Comparison (90→{TASK})",
    fontsize=15
)

plt.xlabel("Future Day")
plt.ylabel("Scaled Power")

plt.grid(alpha=0.3)

plt.legend()

plt.tight_layout()

save_path = os.path.join(
    SAVE_DIR,
    f"prediction_compare_{TASK}.png"
)

plt.savefig(save_path, dpi=300)

plt.show()

print("图片已保存：", save_path)
