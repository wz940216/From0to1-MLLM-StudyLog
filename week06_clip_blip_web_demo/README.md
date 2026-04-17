# week06_clip_blip_web_demo

## 项目简介

`webdemo.py` 是一个基于 Gradio 的图像与文本交互演示项目。集成了 BLIP 图像字幕生成和 CLIP 文本检索两大功能：

- `Image → Caption`：用户上传图片后，自动生成图片描述，并将描述文字覆盖显示在图片左上角。
- `Text → Image`：用户输入文本检索语句后，返回与文本最相似的本地图片，并以图文卡片形式展示。

## 主要功能

1. 图片上传并自动生成 Caption，展示在图片左上角。
2. 文本检索本地图片库，返回最相关的图片结果。
3. 基于 HTML/CSS 自定义输出样式，支持卡片式展示和左上角文字覆盖。

## 技术栈

- Python 3
- Gradio：用于快速构建 Web UI
- PyTorch：模型推理后端
- BLIP：生成图片描述
- CLIP：文本与图片检索匹配
- PIL：图像读取与编码

## 使用方式

1. 切换到项目根目录：

```bash
cd week06_clip_blip_web_demo/code
```

2. 安装依赖（如尚未安装）：

```bash
pip install -r ../../requirements.txt
```

3. 运行演示：

```bash
python webdemo.py
```

4. 在浏览器中打开 Gradio 提供的本地地址，进入演示页面。

## 目录结构

- `code/webdemo.py`：主演示脚本
- `image_index.pt`：CLIP 检索图像特征索引
- `README.md`：本文件
- `*.png` / `*.gif`：示例截图和演示图

## 说明

- `webdemo.py` 中使用了 `week04_clip_retrieval` 的 CLIP 图像检索模块和 `week05_blip_blip2_caption` 的 BLIP字幕生成模块。
- 若要替换图片库或模型路径，可修改 `webdemo.py` 中的 `image_dir` 与 `model_path`。

## 注意

- 运行前请确保 `dataset/COCOCaption/val2017` 图像目录存在且包含 JPEG 图片。
- 运行时第一次会构建索引并保存到 `week06_clip_blip_web_demo/image_index.pt`，后续运行会直接加载索引以加速启动。

## 示例

### Image → Caption

![Image to Caption 示例](./img2txt1.png)

### Text → Image

![Text to Image 示例](./text2image_demo.gif)

