# YOLO 学习笔记

用三个可以运行的 Jupyter Notebook，把 YOLO 从“框、IoU、NMS”一路讲到“手写 YOLOv1”和“拆开 YOLOv8 的 `[1, 84, 8400]`”。

三个 notebook 都已经**带输出执行过**，打开就能看；也可以从头清空重跑。

```text
YOLO/
├── README.md
├── requirements.txt
├── 01_detection_basics.ipynb        # 检测基础：坐标、IoU、NMS、AP、网格、anchor
├── 02_yolov1_from_scratch.ipynb     # 从零实现并训练一个迷你 YOLOv1
├── assets/
│   ├── bus.jpg                      # 示例图片（Ultralytics）
│   ├── yolov8_demo.png              # YOLOv8 检测效果预览
│   └── yolov8n.pt                   # YOLOv8n COCO 预训练权重
└── checkpoints/
    └── yolo1_tiny.pt                # notebook 02 训练出的迷你模型
```

---

## 快速开始

### 1. 安装依赖

```bash
# 建议先按 https://pytorch.org 的说明安装与你显卡匹配的 PyTorch
python -m pip install -r requirements.txt
```

`requirements.txt` 里的 `torch` 不会强制指定 CUDA 版本。如果你要用 GPU，先安装 CUDA 版 PyTorch，再装其余依赖。

### 2. 启动 Jupyter

```bash
jupyter lab
# 或者
jupyter notebook
```

打开任意 `.ipynb`，选择你的 Python 环境作为 kernel，然后从上到下执行。

> notebook 的 metadata 里写的是作者本机的 `yolo-cuda` kernel。如果你的机器上没有这个名字，Jupyter 会提示找不到 kernel，在右上角换成你自己的 Python 3 kernel 即可，不影响代码。

### 3. 只想快速看结果？

直接打开 notebook 看已保存的输出即可。想重跑：

- `01`：约 10 秒，不依赖 GPU。
- `02`：GPU 约 3 分钟；CPU 建议把配置里的 `FAST = True`，数据量和 epoch 都会减少。
- `03`：约 10 秒，需要 `ultralytics` 和本地权重。

---

## 学习路线

| 顺序 | Notebook | 你会搞懂 | 建议时间 |
|---|---|---|---|
| 1 | [检测基础](01_detection_basics.ipynb) | `xyxy`/`cxcywh` 互转、IoU、NMS、Precision/Recall/AP、YOLO 的网格直觉、anchor 与 K-means | 45–60 分钟 |
| 2 | [从零实现 YOLOv1](02_yolov1_from_scratch.ipynb) | 输出张量、目标分配、YOLO loss、解码、NMS、训练与评估 | 60–90 分钟 |

如果时间有限，至少按这个顺序读：

```text
01 的 §2 IoU、§3 NMS、§5 网格
        ↓
02 的 §3 目标分配、§4 loss、§6 解码
```

---

## 一句话理解 YOLO

> 把图像划分成密集的预测位置，每个位置直接回归“这里有没有物体、框在哪、是什么类”，整张图只过一次网络。

三个版本的关键差异：

| | YOLOv1（notebook 02） | YOLOv8 |
|---|---|---|
| 输出 | `7×7×30` | `84×8400`（640 正方形输入） |
| 每个位置 | 2 个框 + objectness + 20 类 | 4 个距离 + 80 类 |
| Anchor | 无 | 无（anchor-free） |
| 正样本分配 | 中心点格子 + IoU 最大的框 | Task-Aligned Assigner |
| 框损失 | MSE | CIoU + DFL |
| 分类损失 | MSE | BCE |
| 多尺度 | 否 | 是（stride 8/16/32） |
| 后处理 | NMS | NMS |

**没有变的骨架**：单阶段、密集预测、端到端训练、一次前向、解码 + NMS。

**变了的血肉**：骨干网络、特征金字塔、标签分配、损失函数、数据增强、工程优化。

---

## 关键数字速查

### YOLOv1

$$
S\times S\times\big(B\times(4+1)+C\big)=7\times7\times(2\times5+20)=1470
$$

### YOLOv8（COCO 80 类，640×640 输入）

```text
80×80 + 40×40 + 20×20 = 6400 + 1600 + 400 = 8400 个候选位置
4 个框参数 + 80 个类别 = 84 个通道
输出：[1, 84, 8400]
每个尺度的原始通道：4×16（DFL）+ 80 = 144
```

注意：`8400` 只对应正方形 640 输入。Ultralytics 默认对长图做 letterbox，例如 `810×1080` 的公交车图会得到 `640×480`，候选数变成 `6300`。

---

## 环境说明

本项目在以下环境验证通过：

| 组件 | 版本 |
|---|---|
| Python | 3.12.10 |
| PyTorch | 2.13.0+cu126 |
| Ultralytics | 8.3.217 |
| NumPy | 2.2.6 |
| Matplotlib | 3.10.7 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |

`02` 在 CPU 上也能跑，但会慢很多；把 `FAST` 设为 `True` 会使用 240 张训练图、8 个 epoch。

---

## 常见问题

### 为什么 `02` 的框比真实 YOLO 少很多？

因为它是一个**教学用迷你模型**：输入只有 112×112，骨干只有 5 层卷积，类别只有 3 个。它的目的不是刷 COCO 指标，而是让你看清每一个张量和每一项 loss。

### 为什么 `8400` 有时候变成 `6300` 或 `1680`？

候选数由**实际网络输入尺寸**决定：

```text
候选数 = (H/8)×(W/8) + (H/16)×(W/16) + (H/32)×(W/32)
```

`8400` 对应 `640×640`；`6300` 对应 `480×640`；`1680` 对应 `320×256` 左右。先看 `result.orig_shape` 和预处理后的输入，不要直接假设。

### 训练 loss 下降但 mAP 不涨？

检测任务常见原因：

- 验证集和训练集分布不同。
- 标签格式错误（最常见：把 `xywh` 的左上角当成了中心点）。
- 类别 id 和 `data.yaml` 的 `names` 顺序不一致。
- 过拟合：数据太少、增强太弱。
- 评估阈值 / NMS 参数不合适。

先可视化 10 张验证图的 GT 和预测，通常比盯着 loss 更快定位问题。

### `torch.cuda.is_available()` 是 False？

说明当前 kernel 用的 Python 环境没有装 CUDA 版 PyTorch，或者 GPU 驱动不匹配。先确认：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

然后在 Jupyter 右上角切换到正确的 kernel。notebook `02`/`03` 会自动回退到 CPU，只是会慢。

---

## 素材与许可

- `assets/bus.jpg` 来自 [Ultralytics](https://github.com/ultralytics/ultralytics) 的示例图片。
- Ultralytics 代码与模型采用 AGPL-3.0 许可；商用前请阅读其许可条款。
- 其余 notebook 文本与代码为本文档配套的教学示例。

---

## 下一步

1. 在自己的数据上按 YOLO 格式组织图片和标签，训练一次 `yolov8n.pt`。
2. 把 `03` 的 `RUN_TRAINING` 改成 `True`，先跑通 `coco8` 的 3-epoch 烟雾测试。
3. 阅读 Ultralytics 的 `v8DetectionLoss` 和 `TaskAlignedAssigner` 源码，对照 `02` 的 `encode_targets`。
4. 试一次导出：`model.export(format="onnx", imgsz=640)`，再用 ONNX Runtime 推理。
5. 把 `02` 的 MSE 框损失换成 CIoU，观察定位质量的变化。
