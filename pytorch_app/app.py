"""
PyTorch MIDI 音乐生成推理应用
基于 PaddlePaddle 训练的模型进行推理
"""

import os
import sys
import torch
import gradio as gr
from typing import Dict, Optional, List
from datetime import datetime

from config import Config
from midi_processor import MIDIProcessor
from mamba_model import MambaMIDIGenerator, create_model
from audio_synthesizer import AudioSynthesizer


class MIDIGeneratorUI:
    def __init__(self, config: Config, model_path: Optional[str] = None):
        self.config = config
        self.processor = MIDIProcessor(config)
        self.synthesizer = AudioSynthesizer(config)
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f'Using GPU: {torch.cuda.device_count()} device(s) available')
        else:
            self.device = torch.device('cpu')
            print('Using CPU')
        
        self.model = self._load_model(model_path)
        
        self.output_dir = os.path.join(config.PROCESSED_DATA_DIR, 'generated')
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _load_model(self, model_path: Optional[str]) -> MambaMIDIGenerator:
        model = create_model(self.config)
        
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            model_state = checkpoint['model_state_dict']
            
            new_state_dict = {}
            for key, value in model_state.items():
                new_key = key.replace('mamba.', 'mamba.', 1)
                new_state_dict[new_key] = value
            
            model.load_state_dict(new_state_dict, strict=False)
            print(f'Loaded model from {model_path}')
        else:
            print('Using untrained model')
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def generate_midi(self, emotion: str, style: str, key: str, mode: str,
                      bpm: int, bars: int, instruments: List[str],
                      temperature: float, top_k: int, top_p: float,
                      verbose: bool = True) -> tuple:
        try:
            import random
            import time
            
            print("\n" + "=" * 60)
            print("🎵 开始生成音乐 (PyTorch)...")
            print("=" * 60)
            print(f"📊 参数: 情绪={emotion}, 风格={style}, 调性={key}{mode}, BPM={bpm}, 小节数={bars}, 乐器={instruments}")
            
            seed = random.randint(0, 2**32 - 1)
            print(f"🎲 随机种子: {seed}")
            
            emotion_idx = self.processor.emotion_to_idx[emotion]
            style_idx = self.processor.style_to_idx[style]
            key_idx = self.processor.key_to_idx[key]
            mode_idx = self.processor.mode_to_idx[mode]
            
            beats_per_bar = 4
            total_beats = bars * beats_per_bar
            
            avg_note_beats = 0.5
            safety_factor = 3.0
            estimated_notes = int(total_beats / avg_note_beats * safety_factor)
            max_length = min(estimated_notes, self.config.MODEL_CONFIG['max_seq_length'])
            
            print(f"📏 目标: {bars}小节 × {beats_per_bar}拍 = {total_beats}拍")
            print(f"📏 预估音符数: ~{estimated_notes}个, 生成长度: {max_length} tokens")
            
            num_tracks = len(instruments)
            print(f"🎼 音轨数: {num_tracks}")
            
            status_messages = []
            status_messages.append(f"🎵 开始生成音乐 (PyTorch)...")
            status_messages.append(f"📊 参数: 情绪={emotion}, 风格={style}, 调性={key}{mode}, BPM={bpm}, 小节数={bars}, 乐器={instruments}")
            status_messages.append(f"🎲 随机种子: {seed}")
            status_messages.append(f"📏 目标: {bars}小节 × {beats_per_bar}拍 = {total_beats}拍, 预估{estimated_notes}个音符")
            status_messages.append(f"🎼 音轨数: {num_tracks}")
            
            with torch.no_grad():
                print(f"\n🔄 [1/4] 开始生成tokens...")
                token_start_time = time.time()
                
                if num_tracks > 1:
                    tokens_dict = self.model.generate_multi_track(
                        emotion=emotion_idx,
                        style=style_idx,
                        key=key_idx,
                        mode=mode_idx,
                        bpm=bpm,
                        num_tracks=num_tracks,
                        max_length=max_length,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        seed=seed
                    )
                    
                    total_tokens = sum(len(tokens) for tokens in tokens_dict.values())
                    token_time = time.time() - token_start_time
                    print(f"✅ Token生成完成!")
                    print(f"   - 生成时间: {token_time:.2f}秒")
                    print(f"   - 总tokens数: {total_tokens}")
                    print(f"   - 生成速度: {total_tokens/token_time:.1f} tokens/秒")
                    
                    status_messages.append(f"\n🔄 [1/4] Token生成完成: {token_time:.2f}秒, {total_tokens} tokens, {total_tokens/token_time:.1f} tokens/秒")
                    
                    print(f"\n🔄 [2/4] 开始处理多音轨MIDI...")
                    midi_start_time = time.time()
                    
                    track_tokens = {}
                    all_single_notes = 0
                    all_chord_starts = 0
                    all_chord_notes = 0
                    
                    for i, (track_id, tokens) in enumerate(tokens_dict.items()):
                        instrument_idx = self.processor.instrument_to_idx[instruments[i]]
                        track_tokens[instrument_idx] = tokens.tolist()
                        
                        chord_start_token_start = self.processor.chord_start_token_start
                        chord_note_token_start = self.processor.chord_note_token_start
                        
                        single_notes = sum(1 for t in tokens if t < chord_start_token_start)
                        chord_starts = sum(1 for t in tokens if chord_start_token_start <= t < chord_note_token_start)
                        chord_note_tokens = sum(1 for t in tokens if t >= chord_note_token_start)
                        
                        all_single_notes += single_notes
                        all_chord_starts += chord_starts
                        all_chord_notes += chord_note_tokens
                        
                        print(f"   - 音轨{i+1}: {len(tokens)} tokens (单音:{single_notes}, 和弦开始:{chord_starts}, 和弦音符:{chord_note_tokens})")
                    
                    print(f"\n   === Token 分布统计 ===")
                    print(f"   - 单音token: {all_single_notes} ({all_single_notes/total_tokens*100:.1f}%)")
                    print(f"   - 和弦开始token: {all_chord_starts} ({all_chord_starts/total_tokens*100:.1f}%)")
                    print(f"   - 和弦音符token: {all_chord_notes} ({all_chord_notes/total_tokens*100:.1f}%)")
                    print(f"   - 理论音符数: {all_single_notes + all_chord_notes}")
                    
                    midi = self.processor.tokens_to_multi_track_midi(track_tokens, tempo=float(bpm), max_bars=bars)
                    midi_time = time.time() - midi_start_time
                    print(f"✅ MIDI处理完成!")
                    print(f"   - 处理时间: {midi_time:.2f}秒")
                    print(f"   - 音轨数: {len(midi.instruments)}")
                    
                    for idx, inst in enumerate(midi.instruments):
                        instrument_name = self.processor.idx_to_instrument.get(inst.program, f'Track {idx}')
                        print(f"   - 音轨{idx+1} ({instrument_name}): {len(inst.notes)} 音符")
                    
                    status_messages.append(f"🔄 [2/4] MIDI处理完成: {midi_time:.2f}秒, {len(midi.instruments)}音轨")
                else:
                    instrument_idx = self.processor.instrument_to_idx[instruments[0]]
                    
                    tokens = self.model.generate(
                        emotion=emotion_idx,
                        style=style_idx,
                        key=key_idx,
                        mode=mode_idx,
                        bpm=bpm,
                        max_length=max_length,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        seed=seed
                    )
                    
                    token_time = time.time() - token_start_time
                    print(f"✅ Token生成完成!")
                    print(f"   - 生成时间: {token_time:.2f}秒")
                    print(f"   - tokens数: {len(tokens)}")
                    print(f"   - 生成速度: {len(tokens)/token_time:.1f} tokens/秒")
                    
                    status_messages.append(f"\n🔄 [1/4] Token生成完成: {token_time:.2f}秒, {len(tokens)} tokens, {len(tokens)/token_time:.1f} tokens/秒")
                    
                    print(f"\n🔄 [2/4] 开始处理MIDI...")
                    midi_start_time = time.time()
                    
                    tokens = tokens.tolist()
                    
                    chord_start_token_start = self.processor.chord_start_token_start
                    chord_note_token_start = self.processor.chord_note_token_start
                    
                    single_note_count = sum(1 for t in tokens if t < chord_start_token_start)
                    chord_count = sum(1 for t in tokens if chord_start_token_start <= t < chord_note_token_start)
                    chord_note_count = sum(1 for t in tokens if t >= chord_note_token_start)
                    
                    print(f"   - 单音token: {single_note_count}")
                    print(f"   - 和弦开始token: {chord_count}")
                    print(f"   - 和弦音符token: {chord_note_count}")
                    
                    if verbose:
                        print(f"\n   前20个token: {tokens[:20]}")
                        print(f"   Token类型分布:")
                        for i, t in enumerate(tokens[:30]):
                            if t < chord_start_token_start:
                                note_idx = t // self.processor.num_durations
                                dur_idx = t % self.processor.num_durations
                                print(f"     [{i}] {t}: 单音 note={note_idx+self.processor.min_note} ({self.processor._idx_to_duration_type(dur_idx)})")
                            elif t < chord_note_token_start:
                                dur_idx = t - chord_start_token_start
                                print(f"     [{i}] {t}: 和弦开始 ({self.processor._idx_to_duration_type(dur_idx)})")
                            else:
                                pitch = t - chord_note_token_start + self.processor.min_note
                                print(f"     [{i}] {t}: 和弦音符 pitch={pitch}")
                    
                    midi = self.processor.tokens_to_midi(tokens, instrument_program=instrument_idx, tempo=float(bpm), max_bars=bars)
                    
                    midi_time = time.time() - midi_start_time
                    note_count = sum(len(inst.notes) for inst in midi.instruments)
                    print(f"✅ MIDI处理完成!")
                    print(f"   - 处理时间: {midi_time:.2f}秒")
                    print(f"   - 音轨数: {len(midi.instruments)}")
                    
                    for idx, inst in enumerate(midi.instruments):
                        instrument_name = self.processor.idx_to_instrument.get(inst.program, f'Track {idx}')
                        print(f"   - 音轨{idx+1} ({instrument_name}): {len(inst.notes)} 音符")
                    
                    status_messages.append(f"🔄 [2/4] MIDI处理完成: {midi_time:.2f}秒, {len(midi.instruments)}音轨")
            
            print(f"\n🔄 [3/4] 开始保存MIDI文件...")
            save_start_time = time.time()
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            midi_filename = f'generated_{timestamp}.mid'
            midi_path = os.path.join(self.output_dir, midi_filename)
            
            self.processor.save_midi(midi, midi_path)
            
            save_time = time.time() - save_start_time
            print(f"✅ MIDI文件保存完成!")
            print(f"   - 保存时间: {save_time:.2f}秒")
            print(f"   - 文件路径: {midi_path}")
            
            status_messages.append(f"🔄 [3/4] MIDI文件保存完成: {save_time:.2f}秒")
            
            print(f"\n🔄 [4/4] 开始合成WAV音频...")
            wav_start_time = time.time()
            
            wav_filename = midi_filename.replace('.mid', '.wav')
            wav_path = os.path.join(self.output_dir, wav_filename)
            
            success = self._synthesize_wav(midi_path, wav_path)
            
            wav_time = time.time() - wav_start_time
            total_time = token_time + midi_time + save_time + wav_time
            
            if success:
                print(f"✅ WAV合成完成!")
                print(f"   - 合成时间: {wav_time:.2f}秒")
                print(f"   - 文件路径: {wav_path}")
                print(f"\n🎉 全部完成! 总耗时: {total_time:.2f}秒")
                print("=" * 60 + "\n")
                
                status_messages.append(f"🔄 [4/4] WAV合成完成: {wav_time:.2f}秒")
                status_messages.append(f"🎉 全部完成! 总耗时: {total_time:.2f}秒")
                
                instruments_str = ', '.join(instruments)
                final_message = '\n'.join(status_messages)
                return midi_path, wav_path, final_message
            else:
                print(f"❌ WAV合成失败!")
                print(f"\n⚠️ MIDI生成成功，但WAV转换失败")
                print("=" * 60 + "\n")
                
                status_messages.append(f"⚠️ MIDI生成成功，但WAV转换失败")
                
                instruments_str = ', '.join(instruments)
                final_message = '\n'.join(status_messages)
                return midi_path, None, final_message
        
        except Exception as e:
            print(f"\n❌ 生成失败: {str(e)}")
            print(f"错误类型: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            print("=" * 60 + "\n")
            
            error_message = f"❌ 生成失败: {str(e)}\n\n错误详情:\n{str(type(e).__name__)}"
            import traceback
            error_message += f"\n\n堆栈跟踪:\n{traceback.format_exc()}"
            return None, None, error_message
    
    def _synthesize_wav(self, midi_path: str, wav_path: str) -> bool:
        """使用 AudioSynthesizer 合成 WAV 音频"""
        return self.synthesizer.midi_to_wav(midi_path, wav_path, sample_rate=44100)
    
    def create_interface(self):
        with gr.Blocks(title="Mamba MIDI 音乐生成器 (PyTorch)") as interface:
            gr.Markdown("# 🎵 Mamba MIDI 音乐生成器 (PyTorch)")
            gr.Markdown("基于 PaddlePaddle 训练模型的 PyTorch 推理版本")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🎨 音乐参数")
                    
                    emotion_dropdown = gr.Dropdown(
                        choices=self.config.EMOTIONS,
                        value="快乐",
                        label="情绪"
                    )
                    
                    style_dropdown = gr.Dropdown(
                        choices=self.config.STYLES,
                        value="流行",
                        label="风格"
                    )
                    
                    with gr.Row():
                        key_dropdown = gr.Dropdown(
                            choices=self.config.KEYS,
                            value="C",
                            label="调性"
                        )
                        
                        mode_dropdown = gr.Dropdown(
                            choices=self.config.MODES,
                            value="major",
                            label="调式"
                        )
                    
                    bpm_slider = gr.Slider(
                        minimum=60,
                        maximum=180,
                        value=120,
                        step=1,
                        label="BPM (速度)"
                    )
                    
                    bars_slider = gr.Slider(
                        minimum=4,
                        maximum=128,
                        value=8,
                        step=1,
                        label="小节数"
                    )
                    
                    gr.Markdown("### ⚙️ 生成参数")
                    
                    temperature_slider = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Temperature (随机性)"
                    )
                    
                    top_k_slider = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=50,
                        step=5,
                        label="Top-K (采样范围)"
                    )
                    
                    top_p_slider = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.9,
                        step=0.05,
                        label="Top-P (核采样)"
                    )
                    
                    generate_btn = gr.Button("🎼 生成音乐", variant="primary", size="lg")
                    
                with gr.Column(scale=1):
                    instruments_checkbox = gr.CheckboxGroup(
                        choices=self.config.INSTRUMENTS,
                        value=["Acoustic Grand Piano"],
                        label="乐器 (可多选)"
                    )
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📄 输出")
                    
                    midi_output = gr.File(
                        label="MIDI 文件",
                        file_types=[".mid", ".midi"]
                    )
                    
                    wav_output = gr.Audio(
                        label="WAV 音频",
                        type="filepath"
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("### 📝 生成信息")
                    
                    info_text = gr.Textbox(
                        label="生成结果",
                        lines=10,
                        interactive=False
                    )
            
            gr.Markdown("### 💡 使用说明")
            gr.Markdown("""
            - **情绪**: 选择歌曲的情感基调（快乐、悲伤、激昂、平静）
            - **风格**: 选择音乐风格（流行、民谣、摇滚、中国风、说唱、R&B、舞曲）
            - **调性**: 选择歌曲的调（C、C#、D等）
            - **调式**: 选择大调或小调
            - **BPM**: 设置每分钟节拍数，影响音乐速度
            - **小节数**: 设置生成的音乐长度
            - **乐器**: 可多选乐器，选择多个乐器将生成多音轨MIDI
            - **Temperature**: 控制生成的随机性，值越高越随机
            - **Top-K/Top-P**: 控制采样策略，影响生成的多样性
            """)
            
            generate_btn.click(
                fn=self.generate_midi,
                inputs=[
                    emotion_dropdown,
                    style_dropdown,
                    key_dropdown,
                    mode_dropdown,
                    bpm_slider,
                    bars_slider,
                    instruments_checkbox,
                    temperature_slider,
                    top_k_slider,
                    top_p_slider
                ],
                outputs=[midi_output, wav_output, info_text]
            )
        
        return interface
    
    def launch(self, share: bool = False, server_port: int = 7865):
        interface = self.create_interface()
        interface.launch(share=share, server_port=server_port)


def main():
    config = Config()
    
    print("=" * 60)
    print("Mamba MIDI 音乐生成器 (PyTorch)")
    print("=" * 60)
    print(f"\n设备: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"输出目录: {os.path.join(config.PROCESSED_DATA_DIR, 'generated')}")
    
    paddle_model_path = os.path.join('..', 'work', 'checkpoints', 'best_model.pdparams')
    torch_model_path = os.path.join('models', 'best_model.pt')
    
    if os.path.exists(torch_model_path):
        print(f"加载 PyTorch 模型: {torch_model_path}")
        model_path = torch_model_path
    elif os.path.exists(paddle_model_path):
        print(f"需要先转换模型: {paddle_model_path}")
        print(f"运行: python convert_model.py")
        model_path = None
    else:
        print("未找到训练好的模型，使用随机初始化模型")
        print("请先运行 PaddlePaddle 版本的 train.py 训练模型")
        model_path = None
    
    print("\n启动Gradio界面...")
    
    app = MIDIGeneratorUI(config, model_path)
    app.launch(server_port=7865)


if __name__ == '__main__':
    main()
