"""
dataset.py

作用：
1. 读取预处理后的 .npy 数据
2. 构建 PyTorch Dataset
3. 构建 DataLoader

"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader




DATA_DIR = "data/processed"

class PowerDataset(Dataset):
    """
    家庭用电预测数据集

    一个样本包括：

    X:
        过去90天所有特征

    y:
        未来90天（或365天）
        global_active_power
    """

    def __init__(self, x_data, y_data):
        """
        x_data : numpy.ndarray
            输入数据

        y_data : numpy.ndarray
            标签数据
        """

        # 转成 float32 Tensor
        self.x = torch.FloatTensor(x_data)
        self.y = torch.FloatTensor(y_data)

    def __len__(self):
        """
        返回数据集样本数量
        """
        return len(self.x)

    def __getitem__(self, index):
        """
        返回一个样本

        返回：
        x.shape = (90, feature_dim)

        y.shape = (90,)
        或
        y.shape = (365,)
        """

        return self.x[index], self.y[index]



# 读取 npy 数据
def load_dataset(task="90"):
    """
    根据任务读取数据

    Parameters
    ----------
    task

    "90"
        90 -> 90

    "365"
        90 -> 365
    """

    if task == "90":

        X_train = np.load(os.path.join(DATA_DIR, "X_train_90.npy"))
        y_train = np.load(os.path.join(DATA_DIR, "y_train_90.npy"))

        X_test = np.load(os.path.join(DATA_DIR, "X_test_90.npy"))
        y_test = np.load(os.path.join(DATA_DIR, "y_test_90.npy"))

    elif task == "365":

        X_train = np.load(os.path.join(DATA_DIR, "X_train_365.npy"))
        y_train = np.load(os.path.join(DATA_DIR, "y_train_365.npy"))

        X_test = np.load(os.path.join(DATA_DIR, "X_test_365.npy"))
        y_test = np.load(os.path.join(DATA_DIR, "y_test_365.npy"))

    else:
        raise ValueError("task只能是'90'或'365'")

    print("=" * 60)
    print(f"当前任务：90 -> {task}")

    print("训练集：")
    print("X:", X_train.shape)
    print("y:", y_train.shape)

    print()

    print("测试集：")
    print("X:", X_test.shape)
    print("y:", y_test.shape)

    print("=" * 60)

    return X_train, y_train, X_test, y_test



# 构建 DataLoader
def build_dataloader(task="90",
                     batch_size=32,
                     shuffle=True):
    """
    构建训练集和测试集 DataLoader
    """

    X_train, y_train, X_test, y_test = load_dataset(task)

    train_dataset = PowerDataset(X_train, y_train)
    test_dataset = PowerDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )

    return train_loader, test_loader



# 测试
if __name__ == "__main__":

    train_loader, test_loader = build_dataloader(
        task="90",
        batch_size=32
    )

    # 查看一个 Batch 的形状
    x, y = next(iter(train_loader))

    print()

    print("一个 Batch：")
    print("X:", x.shape)
    print("y:", y.shape)
