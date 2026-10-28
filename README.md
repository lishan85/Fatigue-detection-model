# 实时人脸疲劳检测（摄像头）

基于 **Ultralytics YOLO 人脸检测** + **MediaPipe Face Mesh** + **OpenCV** 的单脚本演示：从默认摄像头读帧，检测人脸，估计闭眼、打哈欠与头部姿态，并在窗口叠加可视化；满足条件时在画面上显示 **WARNING**。

> **English (short):** A single-file webcam demo that detects faces with a bundled `yolov12l-face.pt` YOLO weights file, extracts MediaPipe face landmarks, computes EAR/MAR/head pose, aggregates signs over 10-second windows, and shows an on-screen fatigue warning. Not a certified driver-monitoring product.

---

## 30 秒速览

| 项目 | 说明 |
|------|------|
| **做什么** | 实时分析摄像头画面中的人脸，提示可能的疲劳行为（闭眼、张嘴、头姿异常等） |
| **怎么跑** | 安装依赖 → `python opencv.py`（可选 `--model`）→ 按 **Esc** 退出 |
| **效果** | OpenCV 窗口「Eye Detection」：眼部/嘴部标注、Pitch/Roll、控制台数值、可选 **WARNING** 大字 |

---

## 功能概览

- **人脸定位**：YOLO（`ultralytics.YOLO`）对整帧推理，置信度阈值 `conf=0.4`。
- **关键点**：在裁剪出的人脸区域上运行 MediaPipe `FaceMesh`（单脸、`refine_landmarks=True`）。
- **指标**（均在 `opencv.py` 内实现）：
  - **EAR**（Eye Aspect Ratio）：左右眼各 6 个 landmark，判断闭眼（阈值 `EAR_THRESHOLD = 0.25`）。
  - **MAR**（Mouth Aspect Ratio）：嘴高/嘴宽，判断张嘴/打哈欠（阈值 `MAR_THRESHOLD = 0.4`）。
  - **Head pose**：`cv2.solvePnP` + 固定 3D 人脸模型点，得到 **Pitch / Roll**（度），用于歪头/点头类判断（`HEAD_PITCH_TH`、`HEAD_ROLL_TH` 均为 `20`）。
- **疲劳逻辑**：每 **10 秒**（`WINDOW_SEC`）统计一次——闭眼帧数 &gt; 30、打哈欠次数 &gt; 3、点头相关帧 &gt; 2 各计 1 分；**≥ 2 分** 则置 `fatigue_warning` 并在画面显示 **WARNING**（警告状态会保持，直到逻辑再次更新）。

**说明：** 仓库当前仅包含运行上述流程所需的 `opencv.py` 与权重 `yolov12l-face.pt`。历史提交中曾出现训练脚本、数据集等文件，**不在当前分支内容中**，本文档不将其列为现网功能。

---

## 技术栈（以代码为准）

| 组件 | 用途 |
|------|------|
| [OpenCV](https://opencv.org/) (`cv2`) | 摄像头、显示、PnP、绘制 |
| [Ultralytics YOLO](https://docs.ultralytics.com/) | 加载 `yolov12l-face.pt` 做人脸检测 |
| [MediaPipe](https://developers.google.com/mediapipe) | `face_mesh` 468 点 landmark |
| NumPy | 距离与矩阵运算 |

**关于 YOLO 版本：** 代码通过 `from ultralytics import YOLO` 加载权重文件名 **`yolov12l-face.pt`**（Large 人脸专用权重）。仓库内**没有** YOLOv11 权重或相关调用；请勿与「YOLOv11」混称，除非您自行更换权重并改代码。

---

## 目录结构

```text
.
├── opencv.py              # 主程序（摄像头循环 + 检测 + 可视化）
├── yolov12l-face.pt       # 人脸检测权重（约 51 MB，已纳入仓库）
├── requirements.txt       # Python 依赖（由 import 推断）
├── .gitignore
└── README.md
```

---

## 环境要求

- **Python**：建议 3.9–3.11（MediaPipe 与 Ultralytics 对 3.12+ 的支持因平台而异，遇安装问题可换 3.10）。
- **硬件**：可用摄像头；GPU 可选（Ultralytics 会自动使用 CUDA，若已安装对应 PyTorch）。
- **系统**：任意支持 OpenCV 摄像头与上述依赖的平台；权重默认与 `opencv.py` 同目录下的 `yolov12l-face.pt`。

---

## 安装

```bash
git clone <your-repo-url>
cd <repo-root>

python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
```

首次运行 Ultralytics 可能还会拉取额外运行时依赖；MediaPipe 会下载其模型资源（需网络）。

---

## 运行

1. **模型路径（默认即可）**  
   程序用 `pathlib` 基于 `opencv.py` 所在目录解析权重，**默认**加载同目录下的 `yolov12l-face.pt`（当前仓库布局下即项目根目录中的权重文件）。  
   若权重在其他位置，可通过命令行覆盖：

   ```bash
   python opencv.py --model /path/to/yolov12l-face.pt
   ```

   `--model` 也可传相对路径，会按当前工作目录解析后再转为绝对路径。

2. **启动**：

   ```bash
   python opencv.py
   ```

3. **交互**：
   - 默认摄像头索引为 `0`（`cv2.VideoCapture(0)`）。
   - 窗口标题：**Eye Detection**。
   - 按 **Esc**（键码 27）退出。
   - 终端会打印 `left_ear`、`right_ear`、`MAR`、`pitch`、`roll` 等调试数值。

---

## 结果展示

仓库内**没有**截图、GIF 或演示视频文件。

<!-- 占位：合并 README 后请在此处补充演示资源，例如 docs/demo.gif -->

**建议后续补充（由维护者添加）：**

- `docs/demo.gif`：正常运行 + 触发 **WARNING** 的短录屏；
- 可选：一两张静态截图（眼部 CLOSED、MOUTH OPEN、WARNING 叠加）。

添加后可在本节改为：

```markdown
![演示 GIF](docs/demo.gif)
```

---

## 可调参数

**命令行**

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `--model` | `<脚本目录>/yolov12l-face.pt` | YOLO 人脸检测权重路径 |

**源码常量**（`opencv.py` 顶部）

| 变量 | 默认值 | 含义 |
|------|--------|------|
| `EAR_THRESHOLD` | 0.25 | EAR 低于此视为闭眼 |
| `MAR_THRESHOLD` | 0.4 | MAR 高于此视为张嘴 |
| `HEAD_PITCH_TH` / `HEAD_ROLL_TH` | 20 | 头部俯仰/侧倾（度） |
| `WINDOW_SEC` | 10 | 疲劳统计窗口（秒） |

人脸检测置信度在 `face_model.predict(..., conf=0.4)` 中写死，需改代码调整。

---

## 已知限制

- **单脚本、配置有限**：除 `--model` 外，摄像头索引、阈值、窗口名等仍在源码中写死。
- **非产品级 DMS**：规则阈值 + 10 秒窗口，未做标定、无多用户、无数据记录；**不能**替代法规要求的驾驶员监控系统。
- **性能**：YOLO 全帧 + MediaPipe 逐人脸裁剪，帧率取决于 CPU/GPU 与分辨率。
- **多人脸**：YOLO 可检多张脸，但 MediaPipe 配置为 `max_num_faces=1`，每张裁剪图内只处理一张 mesh。
- **许可证**：本仓库代码以 [GNU AGPL-3.0](LICENSE) 发布；`yolov12l-face.pt` 权重文件的来源与许可条款见下文「许可证」一节，使用前请自行核实。

---

## 许可证

### 源代码

本仓库中的**源代码**（含 `opencv.py` 等）以 **GNU Affero General Public License v3.0（AGPL-3.0）** 发布。完整许可证正文见仓库根目录 [`LICENSE`](LICENSE)。

选用 AGPL-3.0 的原因之一：本项目依赖 [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)（AGPL-3.0）。若您在网络服务中运行或修改本项目的相关代码，请同时遵守 AGPL-3.0 对源码提供与网络交互条款的要求。

### 模型权重 `yolov12l-face.pt`

仓库内包含约 51 MB 的 **`yolov12l-face.pt`**（YOLO 人脸检测权重，文件名表示 YOLOv12 **L**arge 人脸用途）。该文件自早期提交起即随仓库分发，**提交历史与文档中未记录其原始下载地址或上游发布页**。

- **来源**：**来自网络**（作者本人说明，已无法确认具体出处）。
- **许可**：权重的许可证**未知**，**不受**本仓库 [`LICENSE`](LICENSE) 授权；使用与再分发前请**自行核实**来源和许可。本 README 不对该权重文件授予任何额外权利。

其他依赖（如 OpenCV、MediaPipe、PyTorch 等）各自适用其上游许可证，不在此重复列举。
