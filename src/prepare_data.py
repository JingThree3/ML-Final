import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


RAW_POWER_PATH = "data/raw/household_power_consumption.txt"
RAW_WEATHER_PATH = "data/raw/MENSQ_92_previous-1950-2024.csv"

PROCESSED_DIR = "data/processed"

POWER_DAILY_PATH = os.path.join(PROCESSED_DIR, "power_daily.csv")
MERGED_DAILY_PATH = os.path.join(PROCESSED_DIR, "merged_daily.csv")

TRAIN_CSV_PATH = os.path.join(PROCESSED_DIR, "train.csv")
TEST_CSV_PATH = os.path.join(PROCESSED_DIR, "test.csv")


# 基本参数
INPUT_LEN = 90
SHORT_OUTPUT_LEN = 90
LONG_OUTPUT_LEN = 365

TARGET_COL = "global_active_power"

TRAIN_RATIO = 0.6


def ensure_dir():
    """创建 processed 目录"""
    os.makedirs(PROCESSED_DIR, exist_ok=True)


def load_and_process_power():
    """
    读取家庭用电数据，并按天聚合

    原始数据是分钟级：
    Date + Time + 多个电力变量

    处理：
    global_active_power、global_reactive_power、sub_metering_1、sub_metering_2 按天求和
    voltage、global_intensity 按天求平均
    """
    print("正在读取家庭用电数据...")

    df = pd.read_csv(
        RAW_POWER_PATH,
        sep=";",
        na_values=["?", ""],
        low_memory=False
    )


    df.columns = [col.strip().lower() for col in df.columns]

    # 合并 Date 和 Time，生成真正的时间字段
    df["datetime"] = pd.to_datetime(
        df["date"] + " " + df["time"],
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce"
    )

    # 删除无法解析日期的异常行
    df = df.dropna(subset=["datetime"])

    # 设置时间索引，方便后续按天重采样
    df = df.set_index("datetime").sort_index()

    # 原始 Date 和 Time 已经合并为 datetime，不再需要
    df = df.drop(columns=["date", "time"])

    power_cols = [
        "global_active_power",
        "global_reactive_power",
        "voltage",
        "global_intensity",
        "sub_metering_1",
        "sub_metering_2",
        "sub_metering_3"
    ]

    # 转成数值类型，非法值转为 NaN
    for col in power_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 缺失值处理：先按时间插值，再前后填充
    df[power_cols] = df[power_cols].interpolate(method="time")
    df[power_cols] = df[power_cols].ffill().bfill()

    # 新增第四个分表变量：未被三个分表覆盖的其它用电量
    # global_active_power 单位是 kW，每分钟电量换算为 Wh：kW * 1000 / 60
    df["sub_metering_remainder"] = (
        df["global_active_power"] * 1000 / 60
        - df["sub_metering_1"]
        - df["sub_metering_2"]
        - df["sub_metering_3"]
    )

    # 按天聚合
    daily = pd.DataFrame()

    # 按天求和
    daily["global_active_power"] = df["global_active_power"].resample("D").sum()
    daily["global_reactive_power"] = df["global_reactive_power"].resample("D").sum()
    daily["sub_metering_1"] = df["sub_metering_1"].resample("D").sum()
    daily["sub_metering_2"] = df["sub_metering_2"].resample("D").sum()
    daily["sub_metering_3"] = df["sub_metering_3"].resample("D").sum()
    daily["sub_metering_remainder"] = df["sub_metering_remainder"].resample("D").sum()

    # 按天求平均
    daily["voltage"] = df["voltage"].resample("D").mean()
    daily["global_intensity"] = df["global_intensity"].resample("D").mean()

    # 恢复 date 字段
    daily = daily.reset_index().rename(columns={"datetime": "date"})

    # 再保险处理，防止极少数整天缺失
    daily = daily.ffill().bfill()

    daily.to_csv(POWER_DAILY_PATH, index=False, encoding="utf-8-sig")

    print("家庭用电日级数据已保存：", POWER_DAILY_PATH)
    print(daily.head())

    return daily


def load_and_process_weather():
    """
    读取 Hauts-de-Seine 92 省份的月度气象数据

    RR、NBJRR1、NBJRR5、NBJRR10、NBJBROU

    RR 的单位是毫米的十分之一，所以需要除以 10
    """
    print("正在读取天气数据...")

    weather = pd.read_csv(
        RAW_WEATHER_PATH,
        sep=";",
        low_memory=False
    )

    # 天气数据列名通常是大写，统一处理
    weather.columns = [col.strip().upper() for col in weather.columns]

    need_cols = [
        "AAAAMM",
        "RR",
        "NBJRR1",
        "NBJRR5",
        "NBJRR10",
        "NBJBROU"
    ]

    missing_cols = [col for col in need_cols if col not in weather.columns]
    if missing_cols:
        raise ValueError(f"天气数据缺少字段：{missing_cols}")

    weather = weather[need_cols].copy()

    # AAAAMM 例如 200612，表示 2006年12月
    weather["month"] = pd.to_datetime(
        weather["AAAAMM"].astype(str),
        format="%Y%m",
        errors="coerce"
    )

    weather = weather.dropna(subset=["month"])

    weather_cols = [
        "RR",
        "NBJRR1",
        "NBJRR5",
        "NBJRR10",
        "NBJBROU"
    ]

    for col in weather_cols:
        weather[col] = pd.to_numeric(weather[col], errors="coerce")

    # 同一个省份可能有多个气象站，这里按月份取平均，得到该省份月度气象特征
    weather = weather.groupby("month")[weather_cols].mean().reset_index()

    # RR 单位换算：原始值是 0.1 mm，除以10变成 mm
    weather["RR"] = weather["RR"] / 10.0

    # 缺失值处理
    weather[weather_cols] = weather[weather_cols].ffill().bfill()

    print("天气月度数据处理完成：")
    print(weather.head())

    return weather


def merge_power_and_weather(power_daily, weather_monthly):
    """
    将日级用电数据和月度天气数据融合

    由于天气数据是月度的，所以每一天匹配它所属月份的天气统计值
    """
    print("正在融合用电数据和天气数据...")

    power_daily["date"] = pd.to_datetime(power_daily["date"])

    # 生成月份字段，用于和天气数据连接
    power_daily["month"] = power_daily["date"].values.astype("datetime64[M]")

    merged = power_daily.merge(
        weather_monthly,
        on="month",
        how="left"
    )

    # 删除辅助月份字段
    merged = merged.drop(columns=["month"])

    # 融合后缺失值处理
    merged = merged.ffill().bfill()

    merged.to_csv(MERGED_DAILY_PATH, index=False, encoding="utf-8-sig")

    print("融合后的日级数据已保存：", MERGED_DAILY_PATH)
    print(merged.head())
    print("融合后数据形状：", merged.shape)

    return merged


def split_train_test_by_time(df):
    """
    按时间顺序划分 Train/Test

    """
    print("正在按时间顺序划分 Train/Test...")

    df = df.sort_values("date").reset_index(drop=True)

    split_idx = int(len(df) * TRAIN_RATIO)

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    train_df.to_csv(TRAIN_CSV_PATH, index=False, encoding="utf-8-sig")
    test_df.to_csv(TEST_CSV_PATH, index=False, encoding="utf-8-sig")

    print("训练集已保存：", TRAIN_CSV_PATH, train_df.shape)
    print("测试集已保存：", TEST_CSV_PATH, test_df.shape)

    print("训练集时间范围：", train_df["date"].iloc[0], "到", train_df["date"].iloc[-1])
    print("测试集时间范围：", test_df["date"].iloc[0], "到", test_df["date"].iloc[-1])

    return train_df, test_df


def scale_train_test(train_df, test_df):
    """
    标准化数据。
    """
    feature_cols = [col for col in train_df.columns if col != "date"]

    scaler = StandardScaler()

    train_scaled = train_df.copy()
    test_scaled = test_df.copy()

    train_scaled[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_scaled[feature_cols] = scaler.transform(test_df[feature_cols])

    return train_scaled, test_scaled, feature_cols, scaler


def create_windows(df, feature_cols, input_len, output_len, target_col):
    """
    构造滑动窗口样本。

    X:
    过去 input_len 天的所有特征

    y:
    未来 output_len 天的 global_active_power
    """
    values = df[feature_cols].values.astype(np.float32)

    target_index = feature_cols.index(target_col)

    X = []
    y = []

    total_len = input_len + output_len

    for start_idx in range(len(df) - total_len + 1):
        input_start = start_idx
        input_end = start_idx + input_len

        output_start = input_end
        output_end = input_end + output_len

        x_i = values[input_start:input_end, :]
        y_i = values[output_start:output_end, target_index]

        X.append(x_i)
        y.append(y_i)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    return X, y


def save_window_dataset(train_scaled, test_scaled, feature_cols):
    """
    分别保存 90→90 和 90→365 两套任务的数据
    """
    print("正在构造 90→90 短期预测数据集...")

    X_train_90, y_train_90 = create_windows(
        train_scaled,
        feature_cols,
        INPUT_LEN,
        SHORT_OUTPUT_LEN,
        TARGET_COL
    )

    X_test_90, y_test_90 = create_windows(
        test_scaled,
        feature_cols,
        INPUT_LEN,
        SHORT_OUTPUT_LEN,
        TARGET_COL
    )

    print("正在构造 90→365 长期预测数据集...")

    X_train_365, y_train_365 = create_windows(
        train_scaled,
        feature_cols,
        INPUT_LEN,
        LONG_OUTPUT_LEN,
        TARGET_COL
    )

    X_test_365, y_test_365 = create_windows(
        test_scaled,
        feature_cols,
        INPUT_LEN,
        LONG_OUTPUT_LEN,
        TARGET_COL
    )

    np.save(os.path.join(PROCESSED_DIR, "X_train_90.npy"), X_train_90)
    np.save(os.path.join(PROCESSED_DIR, "y_train_90.npy"), y_train_90)
    np.save(os.path.join(PROCESSED_DIR, "X_test_90.npy"), X_test_90)
    np.save(os.path.join(PROCESSED_DIR, "y_test_90.npy"), y_test_90)

    np.save(os.path.join(PROCESSED_DIR, "X_train_365.npy"), X_train_365)
    np.save(os.path.join(PROCESSED_DIR, "y_train_365.npy"), y_train_365)
    np.save(os.path.join(PROCESSED_DIR, "X_test_365.npy"), X_test_365)
    np.save(os.path.join(PROCESSED_DIR, "y_test_365.npy"), y_test_365)

    # 保存特征名，后面建模时需要知道输入维度和变量顺序
    pd.Series(feature_cols).to_csv(
        os.path.join(PROCESSED_DIR, "feature_cols.csv"),
        index=False,
        header=False,
        encoding="utf-8-sig"
    )

    print("数据集保存完成。")
    print("X_train_90:", X_train_90.shape)
    print("y_train_90:", y_train_90.shape)
    print("X_test_90:", X_test_90.shape)
    print("y_test_90:", y_test_90.shape)

    print("X_train_365:", X_train_365.shape)
    print("y_train_365:", y_train_365.shape)
    print("X_test_365:", X_test_365.shape)
    print("y_test_365:", y_test_365.shape)


def main():
    ensure_dir()

    power_daily = load_and_process_power()
    weather_monthly = load_and_process_weather()

    merged_daily = merge_power_and_weather(
        power_daily,
        weather_monthly
    )

    train_df, test_df = split_train_test_by_time(merged_daily)

    train_scaled, test_scaled, feature_cols, scaler = scale_train_test(
        train_df,
        test_df
    )

    save_window_dataset(
        train_scaled,
        test_scaled,
        feature_cols
    )


if __name__ == "__main__":
    main()
