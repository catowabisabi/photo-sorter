# Photo Sorter

**A local desktop app for organizing personal photo folders by date, visual category, and similar faces.**  
**一个本地桌面照片整理工具，可按日期、视觉类别和相似人脸整理照片。**

![Photo Sorter screenshot](app.png)

## Overview / 项目简介

Photo Sorter is a Windows-friendly Python desktop application built with CustomTkinter. It helps clean up messy photo folders without uploading your images to a cloud service. The app runs locally and uses computer vision models for category sorting and face grouping.

Photo Sorter 是一个适合 Windows 使用的 Python 桌面应用，使用 CustomTkinter 构建。它可以帮助你整理杂乱的照片文件夹，不需要把照片上传到云端。所有处理都在本机运行，并使用视觉模型进行类别分类和人脸分组。

## Key Features / 主要功能

- **Sort by date**: Move root-level images and videos into `YYYY-MM-DD` folders. Files without a detectable date go to `no_date`.
- **按日期整理**：把第一层照片和视频移动到 `YYYY-MM-DD` 文件夹；无法识别日期的文件会进入 `no_date`。

- **Sort by category**: Detect people first, then classify remaining images into user-defined labels such as `food`, `landscape`, and `document`.
- **按类别整理**：先自动识别人物，再把其他图片分到用户定义的类别，例如 `food`、`landscape`、`document`。

- **Group similar faces**: Copy images into `_face_groups/person_001`, `_face_groups/person_002`, and `no_face` using InsightFace embeddings.
- **相似人脸分组**：使用 InsightFace 人脸特征，把照片复制到 `_face_groups/person_001`、`person_002` 和 `no_face`。

- **Local-first workflow**: No cloud upload is required. Your files stay on your computer.
- **本地优先**：不需要云端上传，文件保留在你的电脑上。

## Models / 使用模型

Photo Sorter uses the following models:

Photo Sorter 使用以下模型：

| Purpose | Model |
| --- | --- |
| Person detection / 人物检测 | `facebook/detr-resnet-50` |
| Zero-shot image classification / 零样本图片分类 | `openai/clip-vit-large-patch14` |
| Face recognition / 人脸识别 | `InsightFace buffalo_l` |

For face grouping, only InsightFace detection and recognition modules are loaded. Landmark, age, and gender modules are not required.

人物分组只载入 InsightFace 的 detection 和 recognition 模块，不需要 landmark、年龄或性别模块。

## Installation / 安装

Python 3.11+ is recommended.

建议使用 Python 3.11 或更新版本。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run From Source / 从源码运行

Run the CustomTkinter interface:

运行 CustomTkinter 版本界面：

```powershell
python photo_sorter_ctk.py
```

A basic Tkinter implementation is also available:

项目也保留了一个基础 Tkinter 版本：

```powershell
python photo_sorter_app.py
```

The first run may download model weights from Hugging Face and InsightFace.

首次运行可能会从 Hugging Face 和 InsightFace 下载模型权重。

## Usage / 使用方式

1. Select or paste a target folder path.
2. Optionally edit category labels, for example `food,landscape,document`.
3. Run one of the three actions.

1. 选择或粘贴要整理的资料夹路径。
2. 可按需要修改类别清单，例如 `food,landscape,document`。
3. 点击三个功能之一开始整理。

### Important File Behavior / 重要文件行为

| Action | File behavior |
| --- | --- |
| Sort by date / 按日期整理 | Moves files / 移动文件 |
| Sort by category / 按类别整理 | Moves files / 移动文件 |
| Group similar faces / 相似人脸分组 | Copies files / 复制文件 |

Always test on a copied folder before processing important originals.

处理重要原始照片前，建议先复制一份测试资料夹。

## Build A Windows Release / 构建 Windows 发行版

Install PyInstaller:

安装 PyInstaller：

```powershell
python -m pip install pyinstaller
```

Build the app:

构建应用：

```powershell
pyinstaller --noconfirm PhotoSorterTool.spec
```

The output will be created at:

输出目录：

```text
dist/PhotoSorterTool
```

Keep the full release folder together. Do not distribute only the `.exe` if you bundle libraries or models.

请保留完整发行资料夹。如果你打包了库或模型，不要只发布单独的 `.exe`。

## Bundling Models / 打包模型

This repository does not include model weights.

本仓库不包含模型权重。

For an offline release, place model caches under:

如果需要离线发行版，可把模型缓存放到：

```text
models/huggingface
models/insightface
```

When frozen with PyInstaller, the app checks for bundled model folders next to the app resources.

使用 PyInstaller 打包后，应用会优先查找随程序一起发布的模型资料夹。

## Repository Contents / 仓库内容

```text
photo_sorter_ctk.py      # Main CustomTkinter app / 主界面
photo_sorter_app.py      # Basic Tkinter fallback / 基础 Tkinter 版本
PhotoSorterTool.spec     # PyInstaller build file / 打包配置
requirements.txt         # Python dependencies / Python 依赖
app.png                  # Screenshot used in README / README 截图
```

## License / 许可证

MIT
