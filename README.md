# 实验4 目标检测案例练习 —— 完整代码包

对应实验文档《实验4 目标检测案例练习.docx》，全部脚本在本机验证通过。

## 0. 环境准备

```bash
pip install opencv-python numpy ultralytics
# 说明：OpenCV 必须用 4.x 版本（如 4.12），OpenCV 5.0 已移除 Darknet 导入器
```

已验证环境：
- Python 3.13 / torch 2.14.0+cpu / ultralytics 8.4.197 / opencv 4.12
- CPU: Intel Core i9-14900HX（无显卡）

## 1. 任务1（a）：OpenCV DNN 图片目标检测

```bash
# 1) 下载模型（cfg 和 coco.names 已在 models/ 内，只需 yolov3.weights 约240MB）
python setup_assets.py download-weights --model yolo
python setup_assets.py download-weights --model ssd    # SSD 可选

# 2) 单张图片检测（YOLOv3，CPU 约1.4秒/帧）
python detect_image.py --model yolo --image bus.jpg --conf 0.5

# 批处理/脚本中运行（不弹窗，只保存结果图）：
#   Windows CMD:  set NO_SHOW=1
#   Git Bash:     NO_SHOW=1 python detect_image.py ...
```

输出：终端打印每个目标的类别/置信度/框坐标，结果图保存到 `outputs/`。

## 2. 任务1（b）：调用摄像头实时检测（文档高亮要求）

```bash
python detect_camera.py                    # 默认 YOLOv3 + 摄像头0
python detect_camera.py --record outputs/demo.mp4   # 同时录制
python detect_camera.py --save-every 50    # 每50帧自动截图（实验报告配图）
```

**键盘操作**：
| 键 | 功能 |
|---|---|
| q / ESC | 退出 |
| s | 截图保存到 outputs/ |
| p | 暂停/继续 |
| 空格 | 开关检测（对比原始画面） |
| 1 / 2 | 置信度阈值 -/+ 0.05 |

画面左上角实时显示模型名、FPS、推理耗时、置信度阈值、目标数。
YOLOv3 CPU 下约 1~2 FPS；若嫌卡可换 SSD（`--model ssd`）或 YOLO26n ONNX（`--model onnx`）。

## 3. 任务2（1）：YOLO26 跑通推理

```bash
python yolo26_infer.py --source bus.jpg                 # 单图
python yolo26_infer.py --source 0 --save-video --show   # 摄像头实时
python yolo26_infer.py --source demo.mp4 --save-video   # 视频
python yolo26_infer.py --model yolo26s.pt --source x.jpg  # 换 s 模型
```

首次运行自动下载 yolo26n.pt（5.3MB）。结果图和 result.json 保存到
`outputs/yolo26_infer/`。

已验证：官方示例图 bus.jpg 检出 5 个目标（1 bus + 4 person），
CPU 首帧含模型加载约 1~2 秒，之后单帧约 20~30ms。

## 4. 任务2（2）：数据扩增 + 个人数据集训练

### 4.1 数据扩增（expand_dataset.py）

把老师的少量标注数据（图片 + 同名 .txt YOLO 标签）放进 `raw_images/`，然后：

```bash
# 每张原图生成 6 个变体（翻转/缩放/旋转/平移 + 颜色/噪声/模糊/JPEG伪影）
python expand_dataset.py --src raw_images --dst dataset --copies 6

# 数据极少时启用背景合成（需要准备 backgrounds/ 目录放背景图）
python expand_dataset.py --src raw_images --dst dataset --copies 10 --bg-dir backgrounds --bg-prob 0.3
```

输出 `dataset/images/train`、`dataset/labels/train` 和增强预览图
`augmentation_preview.jpg`（可直接贴实验报告）。
框会随几何变换一起变换并自动裁剪，无效框自动丢弃。

### 4.2 训练（train_yolo26.py）

自动完成：train/val 划分(8:2) -> 标签合法性检查 -> 生成 data.yaml -> 训练 -> 验证。

```bash
# 先快速跑通流程（2~3 epoch，几分钟）
python train_yolo26.py --data dataset --epochs 3 --imgsz 320 --batch 4 --names box

# 正式训练（CPU 约1~3小时，取决于图片数）
python train_yolo26.py --data dataset --epochs 50 --imgsz 416 --batch 4 --names box --export-onnx
```

- `--names`：类别名，逗号分隔，顺序必须与标签里的类别号一致
- 训练产物在 `runs/<实验名>/`：weights/best.pt（最佳权重）、results.png（训练曲线）、confusion_matrix.png 等，全部可直接用于报告
- `--export-onnx`：训练完导出 ONNX 到 models/，可回填给 OpenCV DNN 用

**冒烟测试已验证**：合成数据 60 张（train 48 / val 12），2 epoch 后
mAP50=0.862，Precision=0.916，CPU 推理 16ms/帧。

## 5. 目录结构

```
lab4_yolo_opencv/
├── detect_image.py       # OpenCV DNN 图片检测（yolo/ssd/onnx 三种模型）
├── detect_camera.py      # 摄像头实时检测（HUD/FPS/截图/录制/暂停）
├── yolo26_infer.py       # YOLO26 官方库推理（图/视频/摄像头）
├── expand_dataset.py     # 小数据集扩增（几何+颜色+背景合成，框同步变换）
├── train_yolo26.py       # 训练一条龙（划分/检查/yaml/训练/验证/导出）
├── setup_assets.py       # 模型下载、ONNX导出、演示数据生成
├── models/               # yolov3.weights(248MB)/cfg/coco.names 已就绪
├── dataset/              # 扩增后的训练数据（train 49 / val 12）
├── raw_images/           # 演示原始数据（换成老师数据后重跑上面命令）
├── runs/                 # 训练产物（smoke_test 为冒烟测试结果）
└── outputs/              # 推理结果图/视频
```

## 6. 常见问题

| 问题 | 解决 |
|---|---|
| `readNetFromDarknet` 报错 | OpenCV 5.x 已移除该接口，`pip install "opencv-python==4.12.0.88"` |
| 摄像头打不开 | 关闭占用摄像头的软件（微信/会议/浏览器）；或换 `--camera 1` |
| 摄像头打开要等很久 | 已内置 CAP_DSHOW 后端；仍慢属正常（首次枚举设备） |
| 权重下载失败 | 手动下载 yolov3.weights 放到 models/（pjreddie.com/media/files/yolov3.weights） |
| 训练报 val 为空 | 数据不足时脚本会自动分 1 张进 val；确认 dataset/images/val 有图 |
| CPU 训练太慢 | 减小 --imgsz(320) 和 --epochs，或先 --epochs 3 跑通再通宵跑正式训练 |

## 7. 实验报告素材索引

- YOLOv3 检测结果图：`outputs/yolov3_bus.jpg`
- YOLO26 检测结果图：`outputs/yolo26_infer/bus_yolo26.jpg`
- 数据增强效果预览：`dataset/augmentation_preview.jpg`
- 训练曲线/混淆矩阵：`runs/smoke_test/results.png` 等
- 验证指标：`runs/detect/val/`（mAP50=0.862 的那次评估）
