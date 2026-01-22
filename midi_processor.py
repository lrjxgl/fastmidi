import os
import json
import numpy as np
import pretty_midi
from typing import Dict, List, Tuple, Optional
from config import Config
import paddle


class MIDIProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.midi_config = config.MIDI_CONFIG
        self.min_note = self.midi_config['min_note']
        self.max_note = self.midi_config['max_note']
        self.duration_types = self.midi_config['duration_types']
        
        self.num_notes = self.max_note - self.min_note + 1
        self.num_durations = len(self.duration_types)
        
        self.note_token_start = 0
        self.chord_start_token_start = self.num_notes * self.num_durations
        self.chord_note_token_start = self.chord_start_token_start + self.num_durations
        
        self.vocab_size = self.chord_note_token_start + self.num_notes
        
        self.emotion_to_idx = {emotion: idx for idx, emotion in enumerate(config.EMOTIONS)}
        self.style_to_idx = {style: idx for idx, style in enumerate(config.STYLES)}
        self.key_to_idx = {key: idx for idx, key in enumerate(config.KEYS)}
        self.mode_to_idx = {mode: idx for idx, mode in enumerate(config.MODES)}
        self.instrument_to_idx = {inst: idx for idx, inst in enumerate(config.INSTRUMENTS)}
        
        self.idx_to_emotion = {idx: emotion for emotion, idx in self.emotion_to_idx.items()}
        self.idx_to_style = {idx: style for style, idx in self.style_to_idx.items()}
        self.idx_to_key = {idx: key for key, idx in self.key_to_idx.items()}
        self.idx_to_mode = {idx: mode for mode, idx in self.mode_to_idx.items()}
        self.idx_to_instrument = {idx: inst for inst, idx in self.instrument_to_idx.items()}
    
    def _duration_to_seconds(self, duration_type: str, bpm: float) -> float:
        beat_duration = 60.0 / bpm
        
        duration_map = {
            'whole': 4.0,
            'half': 2.0,
            'quarter': 1.0,
            'eighth': 0.5,
            'sixteenth': 0.25,
            'thirty_second': 0.125
        }
        
        return beat_duration * duration_map.get(duration_type, 1.0)
    
    def load_midi(self, midi_path: str) -> pretty_midi.PrettyMIDI:
        try:
            return pretty_midi.PrettyMIDI(midi_path)
        except Exception as e:
            print(f"Error loading MIDI file {midi_path}: {e}")
            return None
    
    def midi_to_tokens(self, midi: pretty_midi.PrettyMIDI, bpm: float = 120.0) -> List[int]:
        tokens = []
        
        for instrument in midi.instruments:
            if instrument.is_drum:
                continue
            
            notes_by_start = {}
            for note in instrument.notes:
                if note.start < 0 or note.end < 0:
                    continue
                
                if note.pitch < self.min_note or note.pitch > self.max_note:
                    continue
                
                start_time = round(note.start, 6)
                if start_time not in notes_by_start:
                    notes_by_start[start_time] = []
                notes_by_start[start_time].append(note)
            
            sorted_starts = sorted(notes_by_start.keys())
            
            for start_time in sorted_starts:
                notes = notes_by_start[start_time]
                
                if len(notes) == 1:
                    note = notes[0]
                    duration = note.end - note.start
                    duration_idx = self._duration_to_idx(duration, bpm)
                    note_idx = note.pitch - self.min_note
                    token = note_idx * self.num_durations + duration_idx
                    tokens.append(token)
                else:
                    chord_tokens = self._encode_chord(notes, bpm)
                    tokens.extend(chord_tokens)
        
        return tokens
    
    def _encode_chord(self, notes: List, bpm: float) -> List[int]:
        durations = [note.end - note.start for note in notes]
        duration_idx = self._duration_to_idx(durations[0], bpm)
        
        pitches = sorted([note.pitch - self.min_note for note in notes])
        
        chord_start_token = self.chord_start_token_start + duration_idx
        tokens = [chord_start_token]
        
        for pitch in pitches:
            token = self.chord_note_token_start + pitch
            tokens.append(token)
        
        return tokens
    
    def _decode_chord(self, chord_start_token: int, following_tokens: List[int], bpm: float) -> tuple:
        duration_idx = chord_start_token - self.chord_start_token_start
        duration_type = self._idx_to_duration_type(duration_idx)
        duration = self._duration_to_seconds(duration_type, bpm)
        
        pitches = []
        for token in following_tokens:
            if token < self.chord_note_token_start:
                break
            pitch = token - self.chord_note_token_start
            pitches.append(pitch)
        
        return pitches, duration
    
    def _duration_to_idx(self, duration: float, bpm: float) -> int:
        beat_duration = 60.0 / bpm
        
        duration_map = {
            'whole': 4.0,
            'half': 2.0,
            'quarter': 1.0,
            'eighth': 0.5,
            'sixteenth': 0.25,
            'thirty_second': 0.125
        }
        
        best_idx = 0
        min_diff = float('inf')
        
        for idx, (duration_type, multiplier) in enumerate(duration_map.items()):
            expected_duration = beat_duration * multiplier
            diff = abs(duration - expected_duration)
            if diff < min_diff:
                min_diff = diff
                best_idx = idx
        
        return best_idx
    
    def _idx_to_duration_type(self, idx: int) -> str:
        return self.duration_types[idx]
    
    def tokens_to_midi(self, tokens: List[int], 
                      instrument_program: int = 0,
                      tempo: float = 120.0,
                      max_bars: Optional[int] = None) -> pretty_midi.PrettyMIDI:
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        
        instrument_name = self.idx_to_instrument.get(instrument_program, '')
        is_drum = 'Drums' in instrument_name
        
        if is_drum:
            instrument = pretty_midi.Instrument(program=0, is_drum=True)
        else:
            instrument = pretty_midi.Instrument(program=instrument_program, is_drum=False)
        
        current_time = 0.0
        notes = []
        
        if len(tokens) > 1:
            tokens = tokens[1:]
        
        beat_duration = 60.0 / tempo
        beats_per_bar = 4
        max_time = max_bars * beats_per_bar * beat_duration if max_bars else float('inf')
        
        total_single_notes = 0
        total_chord_notes = 0
        skipped_single = 0
        skipped_chord = 0
        skip_reasons = {}
        
        skipped_note_idx_values = {}
        skipped_pitch_values = {}
        exceeded_max_time = 0
        
        i = 0
        while i < len(tokens) and current_time < max_time:
            token = tokens[i]
            
            if token < self.chord_start_token_start:
                duration_idx = token % self.num_durations
                note_idx = token // self.num_durations
                
                pitch = note_idx + self.min_note
                velocity = 80
                duration_type = self._idx_to_duration_type(duration_idx)
                duration = self._duration_to_seconds(duration_type, tempo)
                
                total_single_notes += 1
                
                if is_drum:
                    pitch = min(127, max(0, pitch))
                
                note_start = current_time
                note_end = current_time + duration
                if note_end > max_time:
                    note_end = max_time
                    exceeded_max_time += 1
                
                if note_end > note_start:
                    if pitch < self.min_note or pitch > self.max_note:
                        skipped_single += 1
                        reason = f"pitch超出范围: {pitch} (min:{self.min_note}, max:{self.max_note})"
                        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                        skipped_note_idx_values[note_idx] = skipped_note_idx_values.get(note_idx, 0) + 1
                    else:
                        note = pretty_midi.Note(
                            velocity=velocity,
                            pitch=pitch,
                            start=note_start,
                            end=note_end
                        )
                        notes.append(note)
                else:
                    skipped_single += 1
                    reason = f"note_end({note_end:.3f}) <= note_start({note_start:.3f})"
                    skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                
                current_time += duration
                i += 1
                
                if i < len(tokens) and tokens[i] >= self.chord_note_token_start:
                    skip_reason = f"单个音符后接和弦音符token: {tokens[i]}"
                    skip_reasons[skip_reason] = skip_reasons.get(skip_reason, 0) + 1
                    i += 1
            elif token < self.chord_note_token_start:
                duration_idx = token - self.chord_start_token_start
                
                if duration_idx < 0 or duration_idx >= self.num_durations:
                    i += 1
                    continue
                
                duration_type = self._idx_to_duration_type(duration_idx)
                duration = self._duration_to_seconds(duration_type, tempo)
                velocity = 80
                
                pitches = []
                i += 1
                while i < len(tokens) and tokens[i] >= self.chord_note_token_start:
                    pitch = tokens[i] - self.chord_note_token_start
                    
                    if pitch >= self.num_notes:
                        break
                    
                    pitches.append(pitch)
                    i += 1
                
                if pitches:
                    chord_start = current_time
                    chord_end = current_time + duration
                    if chord_end > max_time:
                        chord_end = max_time
                    
                    print(f"   和弦: 时间 {chord_start:.2f}s-{chord_end:.2f}s, 音符数 {len(pitches)}, 音高 {[p + self.min_note for p in pitches]}")
                    
                    chord_note_count = 0
                    for pitch in pitches:
                        actual_pitch = pitch + self.min_note
                        
                        total_chord_notes += 1
                        
                        if chord_end > chord_start:
                            if is_drum:
                                actual_pitch = min(127, max(0, actual_pitch))
                            
                            if actual_pitch < self.min_note or actual_pitch > self.max_note:
                                skipped_chord += 1
                                reason = f"和弦pitch超出范围: {actual_pitch} (min:{self.min_note}, max:{self.max_note})"
                                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                                skipped_pitch_values[pitch] = skipped_pitch_values.get(pitch, 0) + 1
                            elif chord_end > max_time:
                                skipped_chord += 1
                                exceeded_max_time += 1
                            else:
                                note = pretty_midi.Note(
                                    velocity=velocity,
                                    pitch=actual_pitch,
                                    start=chord_start,
                                    end=chord_end
                                )
                                notes.append(note)
                                chord_note_count += 1
                        else:
                            skipped_chord += 1
                    
                    print(f"      → 成功添加 {chord_note_count}/{len(pitches)} 个音符")
                    
                    current_time += duration
            else:
                i += 1
        
        instrument.notes = notes
        midi.instruments.append(instrument)
        
        print(f"\n   === 解码统计 ===")
        print(f"   单音token总数: {total_single_notes}")
        print(f"   和弦音符token总数: {total_chord_notes}")
        print(f"   实际生成音符: {len(notes)}")
        print(f"   跳过单音: {skipped_single}")
        print(f"   跳过和弦音符: {skipped_chord}")
        print(f"   超出max_time: {exceeded_max_time}")
        
        if skip_reasons:
            print(f"\n   === 跳过原因 (Top 5) ===")
            sorted_reasons = sorted(skip_reasons.items(), key=lambda x: x[1], reverse=True)[:5]
            for reason, count in sorted_reasons:
                print(f"   - {reason}: {count}次")
        
        if skipped_note_idx_values:
            print(f"\n   === 被跳过的单音 note_idx 分布 (Top 10) ===")
            sorted_idx = sorted(skipped_note_idx_values.items(), key=lambda x: x[1], reverse=True)[:10]
            for note_idx, count in sorted_idx:
                pitch = note_idx + self.min_note
                print(f"   - note_idx={note_idx} (pitch={pitch}): {count}次")
        
        if skipped_pitch_values:
            print(f"\n   === 被跳过的和弦 pitch 分布 (Top 10) ===")
            sorted_pitch = sorted(skipped_pitch_values.items(), key=lambda x: x[1], reverse=True)[:10]
            for pitch, count in sorted_pitch:
                actual_pitch = pitch + self.min_note
                print(f"   - pitch偏移={pitch} (实际pitch={actual_pitch}): {count}次")
        
        return midi
    
    def midi_to_multi_track_tokens(self, midi: pretty_midi.PrettyMIDI, 
                                  max_tracks: int = 8,
                                  bpm: float = 120.0) -> Dict[int, List[int]]:
        track_tokens = {}
        
        for instrument in midi.instruments:
            if instrument.is_drum:
                continue
            
            track_id = min(instrument.program, max_tracks - 1)
            
            if track_id not in track_tokens:
                track_tokens[track_id] = []
            
            for note in instrument.notes:
                if note.start < 0 or note.end < 0:
                    continue
                
                if note.pitch < self.min_note or note.pitch > self.max_note:
                    continue
                
                duration = note.end - note.start
                duration_idx = self._duration_to_idx(duration, bpm)
                
                note_idx = note.pitch - self.min_note
                
                token = note_idx * self.num_durations + duration_idx
                track_tokens[track_id].append(token)
        
        return track_tokens
    
    def tokens_to_multi_track_midi(self, track_tokens: Dict[int, List[int]],
                                   tempo: float = 120.0,
                                   max_bars: Optional[int] = None) -> pretty_midi.PrettyMIDI:
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        
        for track_id, tokens in track_tokens.items():
            instrument_name = self.idx_to_instrument.get(track_id, '')
            is_drum = 'Drums' in instrument_name
            
            if is_drum:
                instrument = pretty_midi.Instrument(program=0, is_drum=True)
            else:
                instrument = pretty_midi.Instrument(program=track_id, is_drum=False)
            
            notes = []
            
            if len(tokens) > 1:
                tokens = tokens[1:]
            
            beat_duration = 60.0 / tempo
            beats_per_bar = 4
            max_time = max_bars * beats_per_bar * beat_duration if max_bars else float('inf')
            
            i = 0
            current_time = 0.0
            
            while i < len(tokens) and current_time < max_time:
                token = tokens[i]
                
                if token < self.chord_start_token_start:
                    duration_idx = token % self.num_durations
                    note_idx = token // self.num_durations
                    
                    pitch = note_idx + self.min_note
                    velocity = 80
                    duration_type = self._idx_to_duration_type(duration_idx)
                    duration = self._duration_to_seconds(duration_type, tempo)
                    
                    if is_drum:
                        pitch = min(127, max(0, pitch))
                    
                    note_start = current_time
                    note_end = current_time + duration
                    if note_end > max_time:
                        note_end = max_time
                    
                    if note_end > note_start:
                        note = pretty_midi.Note(
                            velocity=velocity,
                            pitch=pitch,
                            start=note_start,
                            end=note_end
                        )
                        notes.append(note)
                    
                    current_time += duration
                    i += 1
                    
                    if i < len(tokens) and tokens[i] >= self.chord_note_token_start:
                        i += 1
                elif token < self.chord_note_token_start:
                    duration_idx = token - self.chord_start_token_start
                    
                    if duration_idx < 0 or duration_idx >= self.num_durations:
                        i += 1
                        continue
                    
                    duration_type = self._idx_to_duration_type(duration_idx)
                    duration = self._duration_to_seconds(duration_type, tempo)
                    velocity = 80
                    
                    pitches = []
                    i += 1
                    while i < len(tokens) and tokens[i] >= self.chord_note_token_start:
                        pitch = tokens[i] - self.chord_note_token_start
                        
                        if pitch >= self.num_notes:
                            break
                        
                        pitches.append(pitch)
                        i += 1
                    
                    if pitches:
                        chord_start = current_time
                        chord_end = current_time + duration
                        if chord_end > max_time:
                            chord_end = max_time
                        
                        for pitch in pitches:
                            actual_pitch = pitch + self.min_note
                            if is_drum:
                                actual_pitch = min(127, max(0, actual_pitch))
                            
                            if chord_end > chord_start:
                                note = pretty_midi.Note(
                                    velocity=velocity,
                                    pitch=actual_pitch,
                                    start=chord_start,
                                    end=chord_end
                                )
                                notes.append(note)
                        
                        current_time += duration
                else:
                    i += 1
            
            instrument.notes = notes
            midi.instruments.append(instrument)
        
        return midi
    
    def encode_condition(self, emotion: str, style: str, 
                        key: str = 'C', mode: str = 'major',
                        bpm: int = 120) -> Dict[str, int]:
        return {
            'emotion': self.emotion_to_idx.get(emotion, 0),
            'style': self.style_to_idx.get(style, 0),
            'key': self.key_to_idx.get(key, 0),
            'mode': self.mode_to_idx.get(mode, 0),
            'bpm': max(60, min(180, int(bpm)))
        }
    
    def decode_condition(self, condition: Dict[str, int]) -> Dict[str, str]:
        return {
            'emotion': self.idx_to_emotion.get(condition['emotion'], '快乐'),
            'style': self.idx_to_style.get(condition['style'], '流行'),
            'key': self.idx_to_key.get(condition['key'], 'C'),
            'mode': self.idx_to_mode.get(condition['mode'], 'major'),
            'bpm': condition['bpm']
        }
    
    def pad_sequence(self, tokens: List[int], max_length: int, 
                    pad_value: int = 0) -> List[int]:
        if len(tokens) >= max_length:
            return tokens[:max_length]
        return tokens + [pad_value] * (max_length - len(tokens))
    
    def create_training_sample(self, midi_path: str, 
                               tags: Dict[str, str],
                               max_length: int = 2048) -> Optional[Dict]:
        midi = self.load_midi(midi_path)
        if midi is None:
            return None
        
        bpm = float(tags.get('bpm', 120))
        tokens = self.midi_to_tokens(midi, bpm)
        if len(tokens) == 0:
            return None
        
        tokens = self.pad_sequence(tokens, max_length)
        
        condition = self.encode_condition(
            emotion=tags.get('emotion', '快乐'),
            style=tags.get('style', '流行'),
            key=tags.get('key', 'C'),
            mode=tags.get('mode', 'major'),
            bpm=bpm
        )
        
        return {
            'tokens': tokens,
            'condition': condition,
            'length': len(tokens)
        }
    
    def save_midi(self, midi: pretty_midi.PrettyMIDI, output_path: str):
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        midi.write(output_path)
    
    def get_midi_info(self, midi_path: str) -> Optional[Dict]:
        midi = self.load_midi(midi_path)
        if midi is None:
            return None
        
        num_instruments = len([inst for inst in midi.instruments if not inst.is_drum])
        num_notes = sum(len(inst.notes) for inst in midi.instruments if not inst.is_drum)
        
        tempo = 120.0
        if len(midi.time_signature_changes) > 0:
            tempo = midi.estimate_tempo()
        
        return {
            'num_instruments': num_instruments,
            'num_notes': num_notes,
            'tempo': tempo,
            'duration': midi.get_end_time()
        }
    
    def process_dataset(self, tags_file: str = None,
                       max_length: int = 2048,
                       use_cache: bool = True,
                       max_samples: int = None) -> List[Dict]:
        if tags_file is None:
            tags_file = os.path.join(self.config.DATA_ROOT, 'midi_tags.json')
        
        cache_file = os.path.join(self.config.CACHE_DIR, f'processed_data_maxlen{max_length}_max{max_samples}.pkl')
        
        if use_cache and os.path.exists(cache_file):
            print(f'Loading cached data from {cache_file}...')
            import pickle
            with open(cache_file, 'rb') as f:
                samples = pickle.load(f)
            
            if max_samples is not None and len(samples) > max_samples:
                samples = samples[:max_samples]
                print(f'Limited to {max_samples} samples from cache')
            
            return samples
        
        if not os.path.exists(tags_file):
            print(f"Tags file not found: {tags_file}")
            return []
        
        with open(tags_file, 'r', encoding='utf-8') as f:
            tags_data = json.load(f)
        
        samples = []
        
        for relative_path, tags in tags_data.items():
            if max_samples is not None and len(samples) >= max_samples:
                break
                
            midi_path = os.path.join(self.config.RAW_DATA_DIR, relative_path)
            
            if not os.path.exists(midi_path):
                continue
            
            sample = self.create_training_sample(midi_path, tags, max_length)
            if sample is not None:
                sample['midi_path'] = relative_path
                samples.append(sample)
        
        print(f"Processed {len(samples)} samples from {len(tags_data)} tagged files")
        
        if use_cache:
            print(f'Saving cache to {cache_file}...')
            import pickle
            os.makedirs(self.config.CACHE_DIR, exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(samples, f)
        
        return samples


class MIDIDataset(paddle.io.Dataset):
    def __init__(self, samples: List[Dict], processor: MIDIProcessor):
        self.samples = samples
        self.processor = processor
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, paddle.Tensor]:
        sample = self.samples[idx]
        
        tokens = paddle.to_tensor(sample['tokens'], dtype='int64')
        condition = sample['condition']
        
        emotion = paddle.to_tensor(condition['emotion'], dtype='int64')
        style = paddle.to_tensor(condition['style'], dtype='int64')
        key = paddle.to_tensor(condition['key'], dtype='int64')
        mode = paddle.to_tensor(condition['mode'], dtype='int64')
        bpm = paddle.to_tensor(condition['bpm'], dtype='int64')
        
        return {
            'tokens': tokens,
            'emotion': emotion,
            'style': style,
            'key': key,
            'mode': mode,
            'bpm': bpm,
            'length': sample['length']
        }
    
    def collate_fn(self, batch: List[Dict]) -> Dict[str, paddle.Tensor]:
        tokens = paddle.stack([item['tokens'] for item in batch])
        emotion = paddle.stack([item['emotion'] for item in batch])
        style = paddle.stack([item['style'] for item in batch])
        key = paddle.stack([item['key'] for item in batch])
        mode = paddle.stack([item['mode'] for item in batch])
        bpm = paddle.stack([item['bpm'] for item in batch])
        length = paddle.to_tensor([item['length'] for item in batch], dtype='int64')
        
        return {
            'tokens': tokens,
            'emotion': emotion,
            'style': style,
            'key': key,
            'mode': mode,
            'bpm': bpm,
            'length': length
        }
