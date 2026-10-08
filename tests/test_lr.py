# tests/test_lr.py
import sys
from pathlib import Path
# 将项目根目录加入Python模块搜索路径，解决跨文件夹导入src包问题
sys.path.append(str(Path(__file__).parent.parent))
import numpy as np
import matplotlib.pyplot as plt
# 导入两个模型类
from src.gd_linear_reg import GradientDescentLR
from src.least_square_lr import LeastSquareLR

def test_lr_consistency():
    """
    单元测试：校验梯度下降线性回归与最小二乘解析解预测结果一致性
    1. 构造样本数据集
    2. 分别训练最小二乘模型、标准化后的梯度下降模型
    3. 对比两组预测结果，使用assert断言校验误差
    4. 绘制拟合效果图与梯度下降损失收敛曲线
    """
    # 构造训练样本 x自变量，y真实标签
    x = np.array([60, 80, 100, 120, 140, 160, 180])
    y = np.array([70, 90, 115, 130, 150, 175, 190])
    
    # ---------- 最小二乘解析解模型 ----------
    ls = LeastSquareLR()
    ls.fit(x, y)

    # ---------- 梯度下降模型：对特征做标准化处理 ----------
    # 计算x均值和标准差
    x_mean = np.mean(x)
    x_std = np.std(x)
    # Z-score标准化：x_scaled = (x-μ)/σ
    x_scaled = (x - x_mean) / x_std
    # 实例化梯度下降模型，设置学习率、迭代轮数
    gd = GradientDescentLR(lr=0.01, epochs=2000)
    gd.fit(x_scaled, y)

    # 分别用两个模型做预测
    y_pred_ls = ls.predict(x)
    y_pred_gd = gd.predict(x_scaled)

    # 打印预测结果，保留4位小数方便查看对比
    print("最小二乘预测：", np.round(y_pred_ls, 4))
    print("梯度下降预测：", np.round(y_pred_gd, 4))

    # 断言：两组预测值允许误差0.2以内，超出则测试失败
    assert np.allclose(y_pred_gd, y_pred_ls, atol=0.2)
    print("✅测试通过！两个模型预测结果一致")

    # ====================== 绘图模块 ======================
    # 设置Matplotlib中文显示、负号正常渲染
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    # 创建1行2列画布，画布大小14*6
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：样本散点 + 两条拟合直线
    ax1.scatter(x, y, color="blue", label="原始样本", s=60)
    # 生成连续x坐标，用于绘制平滑拟合直线
    x_line = np.linspace(50, 190, 100)
    # 最小二乘模型预测拟合线
    y_line_ls = ls.predict(x_line)
    ax1.plot(x_line, y_line_ls, "r-", lw=2, label="最小二乘拟合")
    # 梯度下降：对绘图用x做同样标准化，再预测
    x_line_scaled = (x_line - x_mean) / x_std
    y_line_gd = gd.predict(x_line_scaled)
    ax1.plot(x_line, y_line_gd, "g--", lw=2, label="梯度下降拟合")

    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("线性回归拟合效果")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 右图：梯度下降每轮迭代损失变化曲线
    ax2.plot(gd.loss_history)
    ax2.set_xlabel("迭代次数 epoch")
    ax2.set_ylabel("MSE损失")
    ax2.set_title("梯度下降损失收敛曲线")
    ax2.grid(alpha=0.3)

    # 自动调整子图间距，防止文字重叠
    plt.tight_layout()
    # 弹出绘图窗口
    plt.show()

if __name__ == "__main__":
    # 直接运行该脚本时，执行单元测试函数
    test_lr_consistency()
