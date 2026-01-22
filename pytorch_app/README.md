# fastmidi midi音乐生成器 (PyTorch 推理版本)

基于 PaddlePaddle 训练模型的 PyTorch 推理实现。

## 项目结构

```
pytorch_app/
├── config.py           # 配置文件
├── mamba_model.py      # Mamba 模型架构 (PyTorch)
├── midi_processor.py   # MIDI 数据处理
├── app.py              # Gradio 推理界面
├── requirements.txt    # Python 依赖
└── models/             # 模型文件目录
    └── best_model.pt   # 转换后的 PyTorch 模型
```

## 安装

```bash
cd pytorch_app
pip install -r requirements.txt
```
 
## 运行推理

```bash
python app.py
```
 
 

## 依赖

- Python 3.10+
- PyTorch 2.0+
- Gradio 4.0+
- pretty_midi
