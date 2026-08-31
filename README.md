# 基于 PyTorch 的 CIFAR-10 图像分类模型训练与评测

## 1. 项目简介
本项目基于 PyTorch 完成 CIFAR-10 图像分类任务，独立实现了从数据集加载、数据批量处理、神经网络构建、模型训练、损失计算、参数优化、测试评估到模型保存与实际推理的完整流程。
通过该项目实践深度学习模型的训练与评测流程，并使用 TensorBoard 对模型训练过程进行可视化分析。

## 2. 技术栈

- Python
- PyTorch
- Torchvision
- TensorBoard
- CNN（卷积神经网络）

## 3. 数据集
使用 CIFAR-10 数据集：
- 训练集：50,000 张
- 测试集：10,000 张
- 图像尺寸：32 × 32
- RGB 三通道
- 共 10 个类别

## 4. 模型结构
模型采用卷积神经网络进行图像特征提取和分类：
输入图像
↓
Conv2d
↓
MaxPool2d
↓
Conv2d
↓
MaxPool2d
↓
Conv2d
↓
MaxPool2d
↓
Flatten
↓
Linear
↓
Linear
↓
10 类分类结果

## 5. 模型训练

- 损失函数：CrossEntropyLoss
- 优化器：SGD
- Batch Size：64
- Epoch：10
- 使用反向传播进行模型参数更新
- 使用 TensorBoard 记录训练过程中的 Loss 和 Accuracy

## 6. 模型评测

训练过程中使用 CIFAR-10 测试集对模型进行评估，主要观察：

- Test Loss
- Test Accuracy

同时保存不同训练阶段的模型参数，并加载训练完成的模型，对本地图片进行实际推理测试。

最终测试集准确率：

**XX%**

## 7. 实验结果
训练过程截图截图
TensorBoard 可视化截图
实际图片推理截图

## 8. 项目收获

通过本项目掌握了深度学习模型从数据输入到最终推理的基本流程：

数据处理 → 模型构建 → 前向计算 → Loss 计算 → 反向传播 → 参数优化 → 模型评测 → 模型保存 → 模型推理
