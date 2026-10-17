# Task 2：猫狗分类

本任务使用 PyTorch 完成猫狗图像分类实验，主要包括卷积核实验、CNN 猫狗分类、数据增强对照、特征图可视化、错误样本分析以及独立图片推理。

详细过程、原理说明和思考题答案见：`notes.md`

---

## 1. 项目内容

本任务主要包含两个部分：

**Practice 1：卷积核实验**

主要完成：

- 图像的 RGB、灰度以及 Tensor 表示；
- NumPy 手写二维卷积；
- 均值模糊；
- 高斯模糊；
- 图像锐化；
- Sobel 边缘检测；
- Valid Padding 和 Same Padding 对比。

**Practice 2：CNN 猫狗分类**

主要完成：

- Dogs vs. Cats 数据集读取与统计；
- 训练集、验证集和测试集划分；
- 图像预处理；
- 随机裁剪、随机水平翻转等数据增强；
- 使用 PyTorch 搭建简单 CNN；
- 记录各层 Tensor Shape；
- 训练 Loss 和 Accuracy 曲线；
- 有数据增强和无数据增强对照实验；
- 第一层和最后一层卷积特征图可视化；
- 错误分类样本分析；
- 保存最佳模型；
- 独立图片推理；
- CNN 模型结构图。

---

## 2. 项目文件

```text
Jotang-ml-task2/
├── images/                     # 实验生成的图片
├── results/                    # 训练得到的模型权重
├── practice1.py                # Practice 1：卷积核实验
├── practice2.py                # Practice 2：CNN 猫狗分类
├── inference.py                # 独立图片推理程序
├── model_structure.py          # 绘制 CNN 模型结构图
├── notes.md                    # 实验记录、结果分析和思考题
└── README.md
```

---

## 3. 数据集

本实验使用Kaggle的 **Dogs vs. Cats** 数据集。

训练集共包含：

- Cat：12500 张
- Dog：12500 张
- 总计：25000 张

由于数据集文件较大，因此没有将原始数据集上传到 GitHub。

**运行 `practice2.py` 时，程序会提示：**

> **请输入 Kaggle Dogs vs. Cats 训练集文件夹路径：**

**此时输入本地存放猫狗图片的文件夹路径即可。**

---

## 4. 说明

**原始 Dogs vs. Cats 数据集未上传至 GitHub。**

实验产生的图片保存在：

```text
Jotang-ml-task2/images/
```

训练得到的模型权重保存在：

```text
Jotang-ml-task2/results/
```

更详细的过程、代码解释、结果分析以及“趁热打铁”问题答案见：`notes.md`

