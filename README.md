# README.md
# 一元线性回归项目（最小二乘法 + 梯度下降）
本项目完整实现了 **一元线性回归** 的两种经典求解方法：
- **最小二乘法（解析解）**：纯数学公式、一步求出最优参数
- **梯度下降法（迭代优化）**：机器学习经典迭代训练方式

项目包含：模型封装、独立实验脚本、单元测试、可视化拟合曲线与损失收敛曲线，是一套完整规范的机器学习入门 Demo。

## 📁 项目结构
```
alg_linear_regression_demo/
├─ experiments/
│  ├─ run_gradient_descent.py  # 单独运行梯度下降实验
│  └─ run_least_square.py      # 单独运行最小二乘实验
├─ src/
│  ├─ __pycache__/             # Python自动生成缓存文件夹（无需手动修改）
│  ├─ __init__.py              # 声明src为Python包，统一向外导出模型类
│  ├─ gd_linear_reg.py         # 梯度下降线性回归模型源码
│  └─ least_square_lr.py       # 最小二乘解析解模型源码
├─ tests/
│  └─ test_lr.py               # 单元测试 + 双模型结果对比 + 绘图可视化
└─ README.md
```
## 📄 文件功能详细说明
### 1. experiments 文件夹（独立实验入口）
用于单独执行算法实验、调参、查看效果，不参与单元校验。
- **run_gradient_descent.py**：加载数据集，训练梯度下降模型，输出权重w、截距b，绘制拟合图与损失收敛曲线。
- **run_least_square.py**：加载数据集，运行最小二乘解析解，输出最优参数，绘制拟合直线。

### 2. src 文件夹（核心模型源码）
存放可复用的模型类，是项目核心代码库。
- **gd_linear_reg.py**
  实现 `GradientDescentLR` 梯度下降一元线性回归
  - 损失函数采用MSE均方误差
  - 自动记录每一轮迭代的损失值，用于绘制收敛曲线
  - 提供`fit()`模型训练方法、`predict()`预测方法

- **least_square_lr.py**
  实现 `LeastSquareLR` 最小二乘解析解一元线性回归
  - 通过数学公式直接求解全局最优w、b，**无迭代**
  - 结果精准，作为基准标准答案，用来校验梯度下降效果
  - 提供`fit()`模型训练方法、`predict()`预测方法

- **__init__.py**
  - 标记src为Python包，统一对外暴露两个模型类
  - 解决导入问题，支持写法：`from src import GradientDescentLR, LeastSquareLR`

> `__pycache__`：Python运行脚本自动生成的字节码缓存目录，用来加速模块加载，**无需修改、删除不影响项目运行**。

### 3. tests 文件夹（单元测试）
- **test_lr.py**
  - 同时实例化最小二乘、梯度下降两个模型，在同一数据集训练
  - 使用`assert`断言自动校验两组模型预测输出是否一致
  - 绘制双图：样本散点+两条拟合对比直线、梯度下降损失收敛曲线
  - 项目总验证入口，修改代码后运行，快速检查模型是否出错

---

## ⚙️ 环境依赖安装
```bash
pip install numpy matplotlib
```

## 🚀 项目运行命令
> 建议在项目根目录 `alg_linear_regression_demo` 下执行PowerShell命令
1. 单元测试 + 双模型对比 + 可视化绘图（推荐主入口）
```powershell
python tests/test_lr.py
```

2. 单独运行梯度下降实验
```powershell
python experiments/run_gradient_descent.py
```

3. 单独运行最小二乘实验
```powershell
python experiments/run_least_square.py
```

---

## 📚 项目原理
### 1. 一元线性回归模型
$$\hat y = w \cdot x + b$$
- $w$：权重（直线斜率）
- $b$：偏置（直线截距）
目标：寻找合适的 $\(w,b\)$，让预测值尽可能贴近真实样本。

### 2. 最小二乘法（解析解）
$$w=\frac{\sum(x-\bar x)(y-\bar y)}{\sum(x-\bar x)^2},\quad b=\bar y - w\bar x$$
- 优点：一步算出全局最优解，无迭代，无超参数，结果稳定
- 缺点：不适合海量大数据场景，无法迁移到深度学习

### 3. 梯度下降（迭代优化）
损失函数MSE（均方误差）：
$$L=\frac{1}{n}\sum_{i=1}^n (y_{pred}-y)^2$$

梯度计算公式：
$$dw=\frac{2}{n}\sum (y_{pred}-y)\cdot x,\quad db=\frac{2}{n}\sum (y_{pred}-y)$$

参数迭代更新：
$$w = w - lr\cdot dw,\quad b = b - lr\cdot db$$
- $lr$：学习率，控制每次参数更新步长
- 不断迭代，持续降低损失函数直到收敛

### 4. 项目调试踩坑与解决方案
1. **梯度下降难以收敛**：原始x取值范围60~180，特征尺度较大。
   ✅ 解决方案：特征标准化 $x_{scaled}=\frac{x-\mu}{\sigma}$，将数据缩至均值0方差1。
2. **标准化后不能直接对比w、b**：标准化改变了参数含义。
   ✅ 正确校验方式：对比模型输出的预测值 $y_{pred}$，而非权重参数。
3. **跨目录运行报 ModuleNotFoundError**：Python搜索路径找不到src包。
   ✅ 解决方案：脚本开头通过`sys.path`动态添加项目根目录。
4. **ImportError无法导入类**：Python不知道src包内的模型类。
   ✅ 解决方案：在`src/__init__.py`显式导出`GradientDescentLR`与`LeastSquareLR`。

---

## ✅ 项目运行效果
- 梯度下降迭代训练完成后，预测结果与最小二乘解析解基本完全一致
- 损失曲线快速下降后趋于平稳，模型成功收敛
- 最小二乘拟合直线与梯度下降拟合直线几乎重合，验证算法等价
- 全部代码无报错，可反复运行，结构规范清晰

---

## 🎓 学习收获
1. 一元线性回归完整数学原理
2. 最小二乘解析解代码实现
3. 梯度下降优化算法、MSE损失函数、学习率、迭代轮数
4. 特征标准化，机器学习基础预处理技术
5. Python工程化项目结构、包管理与模块导入
6. Matplotlib数据可视化：散点图、拟合直线、损失收敛曲线
7. 单元测试`assert`，自动化校验模型正确性
