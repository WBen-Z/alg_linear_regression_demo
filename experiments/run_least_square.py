# experiments/run_least_square.py
import sys
from pathlib import Path
# 获取项目根目录路径，添加到模块搜索路径，解决跨目录导入问题
root = Path(__file__).parent.parent
sys.path.append(str(root))
import numpy as np
import matplotlib.pyplot as plt
# 导入最小二乘线性回归模型类
from src.least_square_lr import LeastSquareLR

# 设置matplotlib中文支持，修复负号显示乱码问题
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 构造数据集，自变量x，真实标签y
x = np.array([60, 80, 100, 120, 140, 160, 180])
y = np.array([70, 90, 115, 130, 150, 175, 190])

# 实例化最小二乘模型并训练
ls_model = LeastSquareLR()
ls_model.fit(x,y)
# 使用训练完成的模型对训练集预测
y_pred = ls_model.predict(x)

# 在控制台输出训练结果
print("===== 最小二乘实验结果 =====")
print(f"训练得到w={ls_model.w:.4f}, b={ls_model.b:.4f}")
print("预测值：", np.round(y_pred,4))

# 绘图：绘制样本点与拟合直线
plt.figure(figsize=(8,6))
# 绘制原始数据散点
plt.scatter(x,y,c="blue",label="原始样本",s=60)
# 生成连续x坐标，用于画出平滑的拟合直线
x_line = np.linspace(50,190,100)
y_line = ls_model.predict(x_line)
plt.plot(x_line,y_line,"r-",lw=2,label="最小二乘拟合直线")

plt.xlabel("x")
plt.ylabel("y")
plt.title("最小二乘拟合效果")
plt.legend()
plt.grid(alpha=0.3)
# 弹出绘图窗口展示图像
plt.show()
