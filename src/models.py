"""
models.py

存放所有深度学习模型

LSTM
Transformer
改进模型
"""

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


if __name__ == "__main__":

    # 假设：
    batch = 32
    seq_len = 90
    feature_dim = 13

    x = torch.randn(batch, seq_len, feature_dim)

    model = LSTMModel(
        input_size=13,
        hidden_size=64,
        num_layers=2,
        output_size=90
    )

    y = model(x)

    print("输入：", x.shape)
    print("输出：", y.shape)
