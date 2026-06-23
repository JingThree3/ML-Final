"""
train_lstm.py

自动五轮实验版：训练 LSTM 基准模型
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

from dataset import build_dataloader
from models import LSTMModel


# ==========================
# 1. 固定随机种子
# ==========================

def set_seed(seed=42):
    """固定随机种子，保证实验尽可能可复现"""
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==========================
# 2. 超参数设置
# ==========================

TASK = "90"          # "90" 或 "365"

SEEDS = [42, 52, 62, 72, 82]

BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3

HIDDEN_SIZE = 64
NUM_LAYERS = 2

INPUT_SIZE = 13
OUTPUT_SIZE = 90 if TASK == "90" else 365

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================
# 3. 输出目录
# ==========================

MODEL_DIR = "outputs/models"
FIGURE_DIR = "outputs/figures"
RESULT_DIR = "outputs/results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ==========================
# 4. 评价函数
# ==========================

def evaluate(model, data_loader, criterion):
    """
    在指定数据集上计算 Loss、MAE、MSE

    当前 MAE/MSE 是在标准化后的 global_active_power 上计算。
    """
    model.eval()

    loss_total = 0.0
    pred_list = []
    true_list = []

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            pred = model(x)
            loss = criterion(pred, y)

            loss_total += loss.item()

            pred_list.append(pred.cpu().numpy())
            true_list.append(y.cpu().numpy())

    avg_loss = loss_total / len(data_loader)

    pred = np.concatenate(pred_list, axis=0)
    true = np.concatenate(true_list, axis=0)

    mae = mean_absolute_error(true.flatten(), pred.flatten())
    mse = mean_squared_error(true.flatten(), pred.flatten())

    return avg_loss, mae, mse, pred, true


# ==========================
# 5. 绘图函数
# ==========================

def plot_loss(train_losses, test_losses, save_path, seed):
    """绘制训练集和测试集 Loss 曲线"""
    plt.figure(figsize=(8, 5))

    plt.plot(train_losses, linewidth=2, label="Train Loss")
    plt.plot(test_losses, linewidth=2, label="Test Loss")

    plt.grid(True, linestyle="--", alpha=0.4)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"LSTM Loss Curve: 90 -> {TASK}, Seed={seed}")

    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_prediction(true, pred, save_path, seed):
    """绘制测试集中第一个样本的预测曲线"""
    sample_idx = 0

    plt.figure(figsize=(10, 5))

    plt.plot(
        true[sample_idx],
        linewidth=2.2,
        label="Ground Truth"
    )

    plt.plot(
        pred[sample_idx],
        linewidth=2.2,
        linestyle="--",
        label="Prediction"
    )

    plt.grid(True, linestyle="--", alpha=0.4)
    plt.xlabel("Future Day")
    plt.ylabel("Scaled Global Active Power")
    plt.title(f"LSTM Prediction Curve: 90 -> {TASK}, Seed={seed}")

    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ==========================
# 6. 单轮实验函数
# ==========================

def run_one_experiment(seed):
    """
    运行一次 LSTM 实验
    """

    set_seed(seed)

    train_loader, test_loader = build_dataloader(
        task=TASK,
        batch_size=BATCH_SIZE
    )

    model = LSTMModel(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=OUTPUT_SIZE
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    best_model_path = os.path.join(
        MODEL_DIR,
        f"best_lstm_{TASK}_seed{seed}.pth"
    )

    loss_fig_path = os.path.join(
        FIGURE_DIR,
        f"lstm_loss_{TASK}_seed{seed}.png"
    )

    pred_fig_path = os.path.join(
        FIGURE_DIR,
        f"lstm_prediction_{TASK}_seed{seed}.png"
    )

    print("=" * 60)
    print(f"开始训练 LSTM，任务：90 -> {TASK}，Seed={seed}")
    print("使用设备：", DEVICE)
    print("=" * 60)

    train_losses = []
    test_losses = []

    best_test_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        train_loss_total = 0.0

        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            pred = model(x)
            loss = criterion(pred, y)

            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()

        train_loss = train_loss_total / len(train_loader)

        test_loss, test_mae, test_mse, _, _ = evaluate(
            model,
            test_loader,
            criterion
        )

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), best_model_path)

        print(
            f"Seed={seed} | "
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss={train_loss:.6f} | "
            f"Test Loss={test_loss:.6f} | "
            f"Test MAE={test_mae:.6f} | "
            f"Test MSE={test_mse:.6f}"
        )

    model.load_state_dict(
        torch.load(best_model_path, map_location=DEVICE)
    )

    final_loss, final_mae, final_mse, pred, true = evaluate(
        model,
        test_loader,
        criterion
    )

    plot_loss(
        train_losses,
        test_losses,
        loss_fig_path,
        seed
    )

    plot_prediction(
        true,
        pred,
        pred_fig_path,
        seed
    )

    print("=" * 60)
    print(f"Seed={seed} 最终测试结果")
    print(f"Best Test Loss : {final_loss:.6f}")
    print(f"MAE            : {final_mae:.6f}")
    print(f"MSE            : {final_mse:.6f}")
    print("最佳模型保存路径：", best_model_path)
    print("Loss 曲线已保存：", loss_fig_path)
    print("预测曲线已保存：", pred_fig_path)
    print("=" * 60)

    return {
        "seed": seed,
        "best_test_loss": final_loss,
        "mae": final_mae,
        "mse": final_mse,
        "best_model_path": best_model_path,
        "loss_fig_path": loss_fig_path,
        "pred_fig_path": pred_fig_path
    }


# ==========================
# 7. 主函数：自动五轮实验
# ==========================

def main():
    all_results = []

    for seed in SEEDS:
        result = run_one_experiment(seed)
        all_results.append(result)

    result_df = pd.DataFrame(all_results)

    result_csv_path = os.path.join(
        RESULT_DIR,
        f"lstm_results_{TASK}.csv"
    )

    result_df.to_csv(
        result_csv_path,
        index=False,
        encoding="utf-8-sig"
    )

    mae_mean = result_df["mae"].mean()
    mae_std = result_df["mae"].std()

    mse_mean = result_df["mse"].mean()
    mse_std = result_df["mse"].std()

    loss_mean = result_df["best_test_loss"].mean()
    loss_std = result_df["best_test_loss"].std()

    print("\n" + "=" * 60)
    print(f"LSTM 五轮实验汇总：90 -> {TASK}")
    print("=" * 60)

    print(result_df[["seed", "best_test_loss", "mae", "mse"]])

    print("-" * 60)
    print(f"Best Test Loss : {loss_mean:.6f} ± {loss_std:.6f}")
    print(f"MAE            : {mae_mean:.6f} ± {mae_std:.6f}")
    print(f"MSE            : {mse_mean:.6f} ± {mse_std:.6f}")
    print("结果表保存路径：", result_csv_path)
    print("=" * 60)


if __name__ == "__main__":
    main()
