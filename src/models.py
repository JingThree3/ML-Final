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

class TransformerModel(nn.Module):
    """
    Transformer基准模型

    输入：
        (batch,90,13)

    输出：
        (batch,90)
        或
        (batch,365)
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

        # 输入映射
        self.input_projection = nn.Linear(
            input_size,
            d_model
        )

        # 位置编码
        self.position_encoding = PositionalEncoding(
            d_model=d_model
        )

        # Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        # Transformer Encoder
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 输出层
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
        输入：

        (batch,90,13)

        输出：

        (batch,output_size)
        """

        # 映射到Transformer维度
        x = self.input_projection(x)

        # 加位置编码
        x = self.position_encoding(x)

        # Transformer Encoder
        x = self.encoder(x)

        # 输出
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
