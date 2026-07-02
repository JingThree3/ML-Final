import torch
import numpy as np
import matplotlib.pyplot as plt

from dataset import build_dataloader
from models import LSTMModel, TransformerModel, WeatherFiLMTransformerModel


# ======================
# 配置
# ======================

TASK = "365"   # "90" 或 "365"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_SIZE = 13
OUTPUT_SIZE = 90 if TASK == "90" else 365


MODEL_PATHS = {
    "LSTM": f"outputs/models/best_lstm_{TASK}_seed42.pth",
    "Transformer": f"outputs/models/best_transformer_{TASK}_seed42.pth",
    "Weather-FiLM": f"outputs/models/best_weather_film_transformer_{TASK}_seed42.pth",
}


# ======================
# 加载模型
# ======================

def load_models():
    lstm = LSTMModel(
        input_size=INPUT_SIZE,
        hidden_size=64,
        num_layers=2,
        output_size=OUTPUT_SIZE
    ).to(DEVICE)

    transformer = TransformerModel(
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE
    ).to(DEVICE)

    weather_model = WeatherFiLMTransformerModel(
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE
    ).to(DEVICE)

    lstm.load_state_dict(torch.load(MODEL_PATHS["LSTM"], map_location=DEVICE))
    transformer.load_state_dict(torch.load(MODEL_PATHS["Transformer"], map_location=DEVICE))
    weather_model.load_state_dict(torch.load(MODEL_PATHS["Weather-FiLM"], map_location=DEVICE))

    lstm.eval()
    transformer.eval()
    weather_model.eval()

    return lstm, transformer, weather_model


# ======================
# 获取数据
# ======================

def get_sample():
    _, test_loader = build_dataloader(task=TASK, batch_size=32)

    for x, y in test_loader:
        return x.to(DEVICE), y.to(DEVICE)


# ======================
# 预测
# ======================

@torch.no_grad()
def predict(model, x):
    return model(x).cpu().numpy()


# ======================
# 主函数
# ======================

def main():

    lstm, transformer, weather = load_models()
    x, y = get_sample()

    # 只取第一个样本
    x0 = x[:1]
    y0 = y[:1].cpu().numpy()[0]

    pred_lstm = predict(lstm, x0)[0]
    pred_trans = predict(transformer, x0)[0]
    pred_weather = predict(weather, x0)[0]

    # ======================
    # 画图
    # ======================
    plt.figure(figsize=(12, 6))

    plt.plot(y0, label="Ground Truth", linewidth=2)

    plt.plot(pred_lstm, "--", label="LSTM")
    plt.plot(pred_trans, "--", label="Transformer")
    plt.plot(pred_weather, "--", label="Weather-FiLM Transformer")

    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.title(f"Prediction Comparison (Task {TASK})")

    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()

    save_path = f"outputs/figures/compare_prediction_{TASK}.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)

    print("图已保存：", save_path)


if __name__ == "__main__":
    main()
