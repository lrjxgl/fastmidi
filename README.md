# fastmidi 高效MIDI音乐生成模型
## https://github.com/lrjxgl/fastmidi

基于 Mamba 架构的 MIDI 音乐生成模型，使用 PaddlePaddle 深度学习框架实现。支持单音轨独奏和多音轨编曲生成，可通过情绪、风格、调性、BPM 等参数控制生成结果。

## 项目特性

- **Mamba 架构**：采用选择性状态空间模型进行序列建模
- **PaddlePaddle 框架**：基于百度飞桨深度学习平台
- **多参数控制**：情绪、风格、调性、调式、BPM、乐器等条件生成
- **多音轨支持**：单音轨独奏和多音轨编曲两种模式
- **FluidSynth 音频合成**：使用 GM 音源合成高质量 WAV 音频
- **ffmpeg 音频编辑**：音频编辑，音频转换
- **Gradio Web 界面**：提供友好的交互界面进行音乐生成

 

## 目录结构

```
fastmidi/
├── work/                      # 工作目录
│   ├── data/                  # 数据目录
│   │   └── processed/         # 处理后的数据
│   ├── checkpoints/           # 模型
├── soundfonts/                # 音源文件
│   └── FluidR3_GM.sf2
├── models/                    # 模型架构文件
├── config.py                  # 配置文件
├── midi_processor.py          # MIDI 数据处理模块
├── mamba_model.py             # Mamba 模型架构
├── app.py                     # Gradio 生成界面
├── audio_synthesizer.py       # 音频合成模块
├── requirements.txt           # Python 依赖
└── README.md                  # 项目文档
```

## 环境要求

- Python 3.10
- PaddlePaddle 2.5+
- CUDA 12.0+（可选，用于 GPU 加速）
- FluidSynth（用于音频合成）

## 安装步骤

### 1. 创建 Conda 环境

```bash
conda create -n fastmidi python=3.10 -y
conda activate fastmidi
```

### 2. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 3. 安装 FluidSynth

**Windows：**
从 https://www.fluidsynth.org/ 下载并安装 FluidSynth，确保 `fluidsynth` 命令在系统 PATH 中。

**Linux：**
```bash
sudo apt-get install fluidsynth
```

**Mac：**
```bash
brew install fluidsynth
```

### 4. 安装 ffmpeg

**Windows：**
从 https://www.gyan.dev/ffmpeg/builds/  下载并安装 ffmpeg，确保 `ffmpeg` 命令在系统 PATH 中。

**Linux：**
```bash
sudo apt-get install ffmpeg
```

**Mac：**
```bash
brew install ffmpeg
```

### 5. 准备音源文件

将 `FluidR3_GM.sf2` 音源文件放置在 `soundfonts/` 目录下。音源文件可从以下地址下载：

https://github.com/gleitz/midi-js-soundfonts

### 6.模型下载 




### 6. 生成音乐

```bash
python app.py
```

启动 Gradio Web 界面，可配置以下参数：

| 参数 | 范围 | 说明 |
|------|------|------|
| 情绪 | 4种 | 歌曲情感基调 |
| 风格 | 7种 | 音乐风格类型 |
| 调性 | 12种 | 歌曲调性 |
| 调式 | 2种 | 大调或小调 |
| BPM | 60-180 | 每分钟节拍数 |
| 小节数 | 4-32 | 生成音乐长度 |
| 乐器 | 128种 | 单音轨模式主乐器 |
| 多音轨生成 | 开关 | 是否生成多音轨 |
| 音轨数量 | 2-8 | 多音轨模式音轨数 |
| Temperature | 0.1-2.0 | 生成随机性控制 |
| Top-K | 0-100 | 采样范围限制 |
| Top-P | 0.1-1.0 | 核采样阈值 |


## 参考资料

- PaddlePaddle：https://www.paddlepaddle.org.cn/
- Mamba 论文：https://arxiv.org/abs/2312.00752
- pretty_midi：https://github.com/craffel/pretty-midi
- FluidSynth：https://www.fluidsynth.org/

## 许可证

本项目仅供学习和研究使用。
