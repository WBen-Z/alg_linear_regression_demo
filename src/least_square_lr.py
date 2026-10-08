# src/least_square_lr.py
import numpy as np

class LeastSquareLR:
    """最小二乘法（解析解）实现一元线性回归"""
    def __init__(self):
        """初始化模型参数"""
        self.w = 0.0  # 权重（直线斜率）
        self.b = 0.0  # 偏置（直线截距）

    def fit(self, x, y):
        """
        模型训练：使用最小二乘解析公式直接求解最优 w、b
        :param x: 训练集自变量，一维numpy数组
        :param y: 训练集真实标签，一维numpy数组
        """
        # 一元线性回归最小二乘公式
        x_mean = np.mean(x)  # 自变量x的均值
        y_mean = np.mean(y)  # 标签y的均值
        numerator = np.sum((x - x_mean) * (y - y_mean))    # 分子部分
        denominator = np.sum((x - x_mean) ** 2)            # 分母部分
        self.w = numerator / denominator                   # 计算斜率w
        self.b = y_mean - self.w * x_mean                  # 计算截距b

    def predict(self, x):
        """
        模型预测
        :param x: 输入自变量
        :return: 预测结果
        """
        return self.w * x + self.b
