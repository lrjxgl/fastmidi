import os
import subprocess
import pretty_midi
import wave
from typing import Optional
from config import Config


class AudioSynthesizer:
    def __init__(self, config: Config):
        self.config = config
        self.soundfont_path = config.SOUNDFONT_PATH
    
    def check_fluidsynth(self) -> bool:
        try:
            result = subprocess.run(['fluidsynth', '-h'], 
                                  capture_output=True)
            return result.returncode == 0
        except (FileNotFoundError, subprocess.SubprocessError):
            return False
    
    def check_soundfont(self) -> bool:
        return os.path.exists(self.soundfont_path)
    
    def get_midi_duration(self, midi_path: str) -> float:
        try:
            midi = pretty_midi.PrettyMIDI(midi_path)
            duration = midi.get_end_time()
            return duration
        except Exception as e:
            print(f"Error getting MIDI duration: {e}")
            return 0.0
    
    def get_wav_duration(self, wav_path: str) -> float:
        try:
            with wave.open(wav_path, 'rb') as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                duration = frames / float(rate)
                return duration
        except Exception as e:
            print(f"Error getting WAV duration: {e}")
            return 0.0
    
    def format_duration(self, duration: float) -> str:
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        return f"{minutes}:{seconds:02d}"
    
    def midi_to_wav(self, midi_path: str, output_wav_path: str, 
                    sample_rate: int = 44100, reverb: bool = False) -> bool:
        if not self.check_fluidsynth():
            print("Error: fluidsynth is not installed or not in PATH")
            return False
        
        if not self.check_soundfont():
            print(f"Error: Soundfont not found at {self.soundfont_path}")
            return False
        
        if not os.path.exists(midi_path):
            print(f"Error: MIDI file not found at {midi_path}")
            return False
        
        midi_duration = self.get_midi_duration(midi_path)
        print(f"MIDI文件时长: {self.format_duration(midi_duration)} ({midi_duration:.2f}秒)")
        
        try:
            cmd = [
                'fluidsynth',
                '-F', output_wav_path,
                '-r', str(sample_rate),
                self.soundfont_path,
                midi_path
            ]
            
            if reverb:
                cmd.insert(1, '-R')
                cmd.insert(2, '1')
            
            os.makedirs(os.path.dirname(output_wav_path), exist_ok=True)
            
            print(f"🔄 正在调用fluidsynth转换音频...")
            
            result = subprocess.run(cmd, capture_output=True, timeout=60)
            
            if result.returncode == 0:
                wav_duration = self.get_wav_duration(output_wav_path)
                print(f"✅ WAV合成完成!")
                print(f"   - WAV音频时长: {self.format_duration(wav_duration)} ({wav_duration:.2f}秒)")
                return True
            else:
                stderr = result.stderr.decode('utf-8', errors='ignore') if result.stderr else ''
                print(f"❌ Error converting MIDI to WAV: {stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ WAV合成超时! (超过60秒)")
            return False
        except Exception as e:
            print(f"❌ Error during audio synthesis: {e}")
            return False
    
    def batch_convert(self, midi_dir: str, output_dir: str, 
                     sample_rate: int = 44100, reverb: bool = False) -> dict:
        results = {'success': 0, 'failed': 0, 'errors': []}
        
        if not os.path.exists(midi_dir):
            print(f"Error: MIDI directory not found at {midi_dir}")
            return results
        
        os.makedirs(output_dir, exist_ok=True)
        
        for filename in os.listdir(midi_dir):
            if filename.endswith('.mid') or filename.endswith('.midi'):
                midi_path = os.path.join(midi_dir, filename)
                wav_filename = filename.replace('.mid', '.wav').replace('.midi', '.wav')
                wav_path = os.path.join(output_dir, wav_filename)
                
                print(f"\n处理文件: {filename}")
                if self.midi_to_wav(midi_path, wav_path, sample_rate, reverb):
                    results['success'] += 1
                else:
                    results['failed'] += 1
                    results['errors'].append(filename)
        
        print(f"\n批量转换完成: {results['success']} 成功, {results['failed']} 失败")
        return results
    
    def midi_to_mp3(self, midi_path: str, output_mp3_path: str, 
                   sample_rate: int = 44100, bitrate: str = '192k') -> bool:
        temp_wav = output_mp3_path.replace('.mp3', '_temp.wav')
        
        if not self.midi_to_wav(midi_path, temp_wav, sample_rate):
            return False
        
        try:
            cmd = [
                'ffmpeg',
                '-i', temp_wav,
                '-b:a', bitrate,
                '-y',
                output_mp3_path
            ]
            
            result = subprocess.run(cmd, capture_output=True)
            
            os.remove(temp_wav)
            
            if result.returncode == 0:
                print(f"Successfully converted {midi_path} to {output_mp3_path}")
                return True
            else:
                stderr = result.stderr.decode('utf-8', errors='ignore') if result.stderr else ''
                print(f"Error converting WAV to MP3: {stderr}")
                return False
                
        except Exception as e:
            print(f"Error during MP3 conversion: {e}")
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
            return False


def main():
    config = Config()
    synthesizer = AudioSynthesizer(config)
    
    print("Checking FluidSynth installation...")
    if synthesizer.check_fluidsynth():
        print("FluidSynth is installed")
    else:
        print("FluidSynth is not installed. Please install it first:")
        print("  - Windows: Download from https://www.fluidsynth.org/")
        print("  - Linux: sudo apt-get install fluidsynth")
        print("  - Mac: brew install fluidsynth")
        return
    
    print(f"\nChecking Soundfont at {config.SOUNDFONT_PATH}...")
    if synthesizer.check_soundfont():
        print("Soundfont found")
    else:
        print("Soundfont not found. Please download FluidR3_GM.sf2 and place it in the soundfonts directory")
        print("Download from: https://schristiancollins.com/generaluser.php")
        return
    
    midi_path = os.path.join(config.PROCESSED_DATA_DIR, 'sample.mid')
    wav_path = os.path.join(config.PROCESSED_DATA_DIR, 'sample.wav')
    
    if os.path.exists(midi_path):
        print(f"\nConverting {midi_path} to WAV...")
        synthesizer.midi_to_wav(midi_path, wav_path)
    else:
        print(f"Sample MIDI file not found at {midi_path}")


if __name__ == '__main__':
    main()
