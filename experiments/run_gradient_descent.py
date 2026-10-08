# experiments/run_gradient_descent.py
import sys
from pathlib import Path
# 自动添加项目根目录，解决ModuleNotFoundError
root = Path(__file__).parent.parent
sys.path.append(str(root))
import numpy as np
import matplotlib.pyplot as plt
# 导入梯度下降线性回归模型类
from src.gd_linear_reg import GradientDescentLR

# 设置matplotlib支持中文显示，解决负号显示异常
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 数据集
x = np.array([60, 80, 100, 120, 140, 160, 180])
y = np.array([70, 90, 115, 130, 150, 175, 190])

# 特征标准化 Z-score
x_mean = np.mean(x)  # 计算x均值
x_std = np.std(x)    # 计算x标准差
x_scaled = (x - x_mean) / x_std

# 初始化训练：实例化梯度下降模型，设置学习率与迭代轮数
gd_model = GradientDescentLR(lr=0.01, epochs=2000)
gd_model.fit(x_scaled, y)
# 使用训练好的模型对训练集做预测
y_pred = gd_model.predict(x_scaled)

# 控制台输出训练结果
print("===== 梯度下降实验结果 =====")
print(f"训练得到w={gd_model.w:.4f}, b={gd_model.b:.4f}")
print("预测值：", np.round(y_pred,4))

# 绘图：创建1行2列画布
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(14,6))
# 左图：原始样本散点 + 梯度下降拟合直线
ax1.scatter(x,y,color="blue",label="原始样本",s=60)
# 生成连续x坐标用于绘制平滑拟合线
x_line = np.linspace(50,190,100)
# 对绘图用x进行同样标准化
x_line_scaled = (x_line - x_mean)/x_std
y_line = gd_model.predict(x_line_scaled)
ax1.plot(x_line,y_line,"g--",lw=2,label="梯度下降拟合")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_title("梯度下降拟合效果")
ax1.legend()
ax1.grid(alpha=0.3)

# 右图：损失收敛曲线，查看MSE随迭代变化
ax2.plot(gd_model.loss_history)
ax2.set_xlabel("epoch迭代次数")
ax2.set_ylabel("MSE损失")
ax2.set_title("损失收敛曲线")
ax2.grid(alpha=0.3)

# 自动调整子图间距，防止文字重叠
plt.tight_layout()
# 展示图像窗口
plt.show()
