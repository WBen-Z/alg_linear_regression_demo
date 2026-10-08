# src/gd_linear_reg.py
import numpy as np

class GradientDescentLR:
    """梯度下降法实现一元线性回归"""
    def __init__(self, lr, epochs):
        """
        初始化梯度下降线性回归模型
        :param lr: 学习率，控制参数每次更新的步长
        :param epochs: 迭代总轮数
        """
        self.lr = lr          # 学习率
        self.epochs = epochs  # 迭代轮数
        self.w = 0.0          # 权重（斜率）初始值
        self.b = 0.0          # 偏置（截距）初始值
        self.loss_history = []  # 记录每一轮MSE损失，用于绘制损失收敛曲线

    def fit(self, x, y):
        """
        模型训练：使用梯度下降迭代更新w和b
        :param x: 训练集自变量，一维numpy数组（已标准化）
        :param y: 训练集真实标签，一维numpy数组
        """
        n = len(x)  # 样本数量
        for _ in range(self.epochs):
            # 前向计算：得到当前参数下的预测值
            y_pred = self.w * x + self.b
            # 计算损失函数对w、b的梯度
            dw = (2 / n) * np.sum((y_pred - y) * x)
            db = (2 / n) * np.sum(y_pred - y)
            # 梯度下降，更新权重和偏置
            self.w -= self.lr * dw
            self.b -= self.lr * db
            # 计算当前轮的MSE均方误差，并保存到历史列表
            mse = np.sum((y_pred - y) ** 2) / n
            self.loss_history.append(mse)

    def predict(self, x):
        """
        模型预测
        :param x: 输入自变量（需要和训练时保持相同标准化处理）
        :return: 预测结果
        """
        return self.w * x + self.b
