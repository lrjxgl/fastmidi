import os
import json
import numpy as np
import pretty_midi
from typing import Dict, List, Tuple, Optional
from config import Config


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
    
    def load_midi(self, midi_path: str) -> Optional[pretty_midi.PrettyMIDI]:
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
                      max_bars: Optional[int] = None,
                      verbose: bool = False) -> pretty_midi.PrettyMIDI:
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
                
                if note_end > note_start:
                    if pitch < self.min_note or pitch > self.max_note:
                        skipped_single += 1
                    else:
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
                        
                        total_chord_notes += 1
                        
                        if chord_end > chord_start:
                            if is_drum:
                                actual_pitch = min(127, max(0, actual_pitch))
                            
                            if actual_pitch < self.min_note or actual_pitch > self.max_note:
                                skipped_chord += 1
                            elif chord_end > max_time:
                                skipped_chord += 1
                            else:
                                note = pretty_midi.Note(
                                    velocity=velocity,
                                    pitch=actual_pitch,
                                    start=chord_start,
                                    end=chord_end
                                )
                                notes.append(note)
                        else:
                            skipped_chord += 1
                    
                    current_time += duration
            else:
                i += 1
        
        instrument.notes = notes
        midi.instruments.append(instrument)
        
        if verbose:
            print(f"\n   === 解码统计 ===")
            print(f"   单音token总数: {total_single_notes}")
            print(f"   和弦音符token总数: {total_chord_notes}")
            print(f"   实际生成音符: {len(notes)}")
            print(f"   跳过单音: {skipped_single}")
            print(f"   跳过和弦音符: {skipped_chord}")
        
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
                                   max_bars: Optional[int] = None,
                                   verbose: bool = False) -> pretty_midi.PrettyMIDI:
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        
        for track_id, tokens in track_tokens.items():
            instrument_name = self.idx_to_instrument.get(track_id, '')
            is_drum = 'Drums' in instrument_name
            
            if is_drum:
                instrument = pretty_midi.Instrument(program=0, is_drum=True)
            else:
                instrument = pretty_midi.Instrument(program=track_id, is_drum=False)
            
            current_time = 0.0
            notes = []
            
            if len(tokens) > 1:
                tokens = tokens[1:]
            
            beat_duration = 60.0 / tempo
            beats_per_bar = 4
            max_time = max_bars * beats_per_bar * beat_duration if max_bars else float('inf')
            
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
                    
                    if is_drum:
                        pitch = min(127, max(0, pitch))
                    
                    note_start = current_time
                    note_end = current_time + duration
                    if note_end > max_time:
                        note_end = max_time
                    
                    if note_end > note_start and self.min_note <= pitch <= self.max_note:
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
                            
                            if chord_end > chord_start and self.min_note <= actual_pitch <= self.max_note:
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
            
            if verbose:
                print(f"   音轨 {track_id} ({instrument_name}): {len(notes)} 音符")
        
        return midi
    
    def save_midi(self, midi: pretty_midi.PrettyMIDI, midi_path: str):
        os.makedirs(os.path.dirname(midi_path), exist_ok=True)
        midi.write(midi_path)
        print(f"   MIDI文件已保存: {midi_path}")


class MIDIDataset:
    def __init__(self, config: Config, split: str = 'train'):
        self.config = config
        self.processor = MIDIProcessor(config)
        
        self.data_dir = config.PROCESSED_DATA_DIR
        self.split = split
        
        self.data_files = self._get_data_files()
    
    def _get_data_files(self) -> List[str]:
        split_file = os.path.join(self.data_dir, f'{self.split}_data.json')
        
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                return json.load(f)
        
        midi_dir = os.path.join(self.data_dir, 'raw')
        if not os.path.exists(midi_dir):
            return []
        
        data_files = []
        for root, dirs, files in os.walk(midi_dir):
            for file in files:
                if file.endswith(('.mid', '.midi')):
                    data_files.append(os.path.join(root, file))
        
        return data_files
    
    def __len__(self) -> int:
        return len(self.data_files)
    
    def __getitem__(self, idx: int) -> Dict:
        midi_path = self.data_files[idx]
        midi = self.processor.load_midi(midi_path)
        
        if midi is None:
            return {'tokens': [], 'emotion': 0, 'style': 0, 'key': 0, 'mode': 0, 'bpm': 120}
        
        bpm = 120
        if midi.initial_tempo:
            bpm = midi.initial_tempo
        
        tokens = self.processor.midi_to_tokens(midi, bpm)
        
        return {
            'tokens': tokens,
            'emotion': 0,
            'style': 0,
            'key': 0,
            'mode': 0,
            'bpm': bpm
        }
