"""
models.py

存放所有深度学习模型

LSTM
Transformer
改进模型
"""
import math

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    LSTM基准模型

    输入：
        (batch_size, 90, feature_dim)

    输出：
        (batch_size, output_len)

    其中：
        output_len = 90 或 365
    """

    def __init__(
            self,
            input_size,
            hidden_size,
            num_layers,
            output_size,
            dropout=0.2
    ):
        super().__init__()

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # 全连接层
        self.fc = nn.Linear(
            hidden_size,
            output_size
        )

    def forward(self, x):
        """
        前向传播

        输入：
            x
            (batch,90,13)

        返回：
            (batch,output_size)
        """

        # LSTM输出
        output, (hidden, cell) = self.lstm(x)

        # hidden形状：
        # (num_layers,batch,hidden_size)

        # 取最后一层隐藏状态
        hidden = hidden[-1]

        # 全连接
        out = self.fc(hidden)

        return out



class PositionalEncoding(nn.Module):
    """
    Transformer位置编码

    使用经典Sin/Cos位置编码
    """

    def __init__(self, d_model, max_len=500):
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(
            0,
            max_len,
            dtype=torch.float
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2
            ).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        # 注册为buffer，不参与训练
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x:
        (batch, seq_len, d_model)
        """

        seq_len = x.size(1)

        return x + self.pe[:, :seq_len]
class WeatherAdaptiveGate(nn.Module):
    """
    天气自适应门控模块

    作用：
    根据当前天气变量本身，动态判断每个天气变量的重要性。
    """

    def __init__(self, weather_dim, hidden_dim=32):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(weather_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, weather_dim),
            nn.Sigmoid()
        )

    def forward(self, weather_x):
        """
        weather_x:
            (batch, seq_len, weather_dim)

        返回：
            gated_weather:
                加权后的天气特征

            gate_weight:
                天气变量权重
        """

        gate_weight = self.gate(weather_x)

        gated_weather = weather_x * gate_weight

        return gated_weather, gate_weight


class FiLMModulation(nn.Module):
    """
    FiLM 条件调制模块

    作用：
    利用天气特征生成 gamma 和 beta，
    对电力特征进行动态调制。

    power' = gamma * power + beta
    """

    def __init__(self, weather_dim, power_dim, hidden_dim=32):
        super().__init__()

        self.gamma_net = nn.Sequential(
            nn.Linear(weather_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, power_dim)
        )

        self.beta_net = nn.Sequential(
            nn.Linear(weather_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, power_dim)
        )

    def forward(self, power_x, weather_x):
        """
        power_x:
            (batch, seq_len, power_dim)

        weather_x:
            (batch, seq_len, weather_dim)
        """

        # 为了训练稳定，不让 gamma 变化过大
        gamma = 1.0 + 0.1 * torch.tanh(self.gamma_net(weather_x))

        beta = 0.1 * torch.tanh(self.beta_net(weather_x))

        modulated_power = gamma * power_x + beta

        return modulated_power


class TransformerModel(nn.Module):
    """
    Weather-FiLM Transformer 改进模型

    输入：
        (batch, 90, 13)

    特征划分：
        前8维：电力特征
        后5维：天气特征

    输出：
        (batch, output_size)
    """

    def __init__(
            self,
            input_size,
            output_size,
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
            dropout=0.1
    ):
        super().__init__()

        self.power_dim = 8
        self.weather_dim = 5

        assert input_size == self.power_dim + self.weather_dim

        # 创新点1：天气自适应门控
        self.weather_gate = WeatherAdaptiveGate(
            weather_dim=self.weather_dim,
            hidden_dim=32
        )

        # 创新点2：FiLM 天气条件调制
        self.film = FiLMModulation(
            weather_dim=self.weather_dim,
            power_dim=self.power_dim,
            hidden_dim=32
        )

        # 标准 Transformer 输入映射
        self.input_projection = nn.Linear(
            input_size,
            d_model
        )

        self.position_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=90
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.fc = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                90 * d_model,
                256
            ),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(
                256,
                output_size
            )
        )

    def forward(self, x):
        """
        x:
            (batch, 90, 13)
        """

        # 前8维是电力变量
        power_x = x[:, :, :self.power_dim]

        # 后5维是天气变量
        weather_x = x[:, :, self.power_dim:]

        # Weather Adaptive Gate
        gated_weather, gate_weight = self.weather_gate(weather_x)

        # FiLM：用天气调制电力特征
        modulated_power = self.film(
            power_x,
            gated_weather
        )

        # 拼回13维特征
        x = torch.cat(
            [
                modulated_power,
                gated_weather
            ],
            dim=-1
        )

        # 标准 Transformer
        x = self.input_projection(x)

        x = self.position_encoding(x)

        x = self.encoder(x)

        out = self.fc(x)

        return out


if __name__ == "__main__":

    batch = 32
    seq_len = 90
    feature_dim = 13

    x = torch.randn(batch, seq_len, feature_dim)

    print("=" * 50)
    print("LSTM")
    print("=" * 50)

    lstm = LSTMModel(
        input_size=13,
        hidden_size=64,
        num_layers=2,
        output_size=90
    )

    y = lstm(x)

    print("输入：", x.shape)
    print("输出：", y.shape)

    print()

    print("=" * 50)
    print("Transformer")
    print("=" * 50)

    transformer = TransformerModel(
        input_size=13,
        output_size=90
    )

    y = transformer(x)

    print("输入：", x.shape)
    print("输出：", y.shape)
