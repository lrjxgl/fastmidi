import os

class Config:
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    WORK_DIR = os.path.join(PROJECT_ROOT, 'work')
    DATA_ROOT = os.path.join(WORK_DIR, 'data')
    RAW_DATA_DIR = os.path.join(DATA_ROOT, 'raw')
    PROCESSED_DATA_DIR = os.path.join(DATA_ROOT, 'processed')
    MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
    CHECKPOINTS_DIR = os.path.join(WORK_DIR, 'checkpoints')
    SOUNDFONTS_DIR = os.path.join(PROJECT_ROOT, 'soundfonts')
    LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')
    CACHE_DIR = os.path.join(WORK_DIR, 'cache')
    
    SOUNDFONT_PATH = os.path.join(SOUNDFONTS_DIR, 'FluidR3_GM.sf2')
    
    MODEL_CONFIG = {
        'd_model': 512,
        'd_state': 16,
        'd_conv': 4,
        'expand': 2,
        'n_layers': 6,
        'dropout': 0.2,
        'max_seq_length': 1024,
        'vocab_size': 622
    }
    
    MIDI_CONFIG = {
        'min_note': 21,
        'max_note': 108,
        'duration_types': ['whole', 'half', 'quarter', 'eighth', 'sixteenth', 'thirty_second']
    }
    
    EMOTIONS = ['快乐', '悲伤', '激昂', '平静']
    KEYS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    MODES = ['major', 'minor']
    INSTRUMENTS = [
        'Acoustic Grand Piano', 'Bright Acoustic Piano', 'Electric Grand Piano', 'Honky-tonk Piano',
        'Electric Piano 1', 'Electric Piano 2', 'Harpsichord', 'Clavi', 'Celesta', 'Glockenspiel',
        'Music Box', 'Vibraphone', 'Marimba', 'Xylophone', 'Tubular Bells', 'Dulcimer',
        'Drawbar Organ', 'Percussive Organ', 'Rock Organ', 'Church Organ', 'Reed Organ',
        'Accordion', 'Harmonica', 'Tango Accordion', 'Acoustic Guitar (nylon)', 'Acoustic Guitar (steel)',
        'Electric Guitar (jazz)', 'Electric Guitar (clean)', 'Electric Guitar (muted)', 'Overdriven Guitar',
        'Distortion Guitar', 'Guitar Harmonics', 'Acoustic Bass', 'Electric Bass (finger)',
        'Electric Bass (pick)', 'Fretless Bass', 'Slap Bass 1', 'Slap Bass 2', 'Synth Bass 1',
        'Synth Bass 2', 'Violin', 'Viola', 'Cello', 'Contrabass', 'Tremolo Strings',
        'Pizzicato Strings', 'Orchestral Harp', 'Timpani', 'String Ensemble 1', 'String Ensemble 2',
        'Synth Strings 1', 'Synth Strings 2', 'Choir Aahs', 'Voice Oohs', 'Synth Voice',
        'Orchestra Hit', 'Trumpet', 'Trombone', 'Tuba', 'Muted Trumpet', 'French Horn',
        'Brass Section', 'Synth Brass 1', 'Synth Brass 2', 'Soprano Sax', 'Alto Sax',
        'Tenor Sax', 'Baritone Sax', 'Oboe', 'English Horn', 'Bassoon', 'Clarinet',
        'Piccolo', 'Flute', 'Recorder', 'Pan Flute', 'Blown Bottle', 'Shakuhachi',
        'Whistle', 'Ocarina', 'Lead 1 (square)', 'Lead 2 (sawtooth)', 'Lead 3 (calliope)',
        'Lead 4 (chiff)', 'Lead 5 (charang)', 'Lead 6 (voice)', 'Lead 7 (fifths)', 'Lead 8 (bass + lead)',
        'Pad 1 (new age)', 'Pad 2 (warm)', 'Pad 3 (polysynth)', 'Pad 4 (choir)', 'Pad 5 (bowed)',
        'Pad 6 (metallic)', 'Pad 7 (halo)', 'Pad 8 (sweep)', 'FX 1 (rain)', 'FX 2 (soundtrack)',
        'FX 3 (crystal)', 'FX 4 (atmosphere)', 'FX 5 (brightness)', 'FX 6 (goblins)', 'FX 7 (echoes)',
        'FX 8 (sci-fi)', 'Sitar', 'Banjo', 'Shamisen', 'Koto', 'Kalimba', 'Bagpipe',
        'Fiddle', 'Shanai', 'Tinkle Bell', 'Agogo', 'Steel Drums', 'Woodblock', 'Taiko Drum',
        'Melodic Tom', 'Synth Drum', 'Reverse Cymbal', 'Guitar Fret Noise', 'Breath Noise',
        'Seashore', 'Bird Tweet', 'Telephone Ring', 'Helicopter', 'Applause', 'Gunshot',
        'Drums'
    ]
    
    INSTRUMENT_PROGRAM_MAP = {
        'Acoustic Grand Piano': 0, 'Bright Acoustic Piano': 1, 'Electric Grand Piano': 2,
        'Honky-tonk Piano': 3, 'Electric Piano 1': 4, 'Electric Piano 2': 5,
        'Harpsichord': 6, 'Clavi': 7, 'Celesta': 8, 'Glockenspiel': 9, 'Music Box': 10,
        'Vibraphone': 11, 'Marimba': 12, 'Xylophone': 13, 'Tubular Bells': 14, 'Dulcimer': 15,
        'Drawbar Organ': 16, 'Percussive Organ': 17, 'Rock Organ': 18, 'Church Organ': 19,
        'Reed Organ': 20, 'Accordion': 21, 'Harmonica': 22, 'Tango Accordion': 23,
        'Acoustic Guitar (nylon)': 24, 'Acoustic Guitar (steel)': 25, 'Electric Guitar (jazz)': 26,
        'Electric Guitar (clean)': 27, 'Electric Guitar (muted)': 28, 'Overdriven Guitar': 29,
        'Distortion Guitar': 30, 'Guitar Harmonics': 31, 'Acoustic Bass': 32,
        'Electric Bass (finger)': 33, 'Electric Bass (pick)': 34, 'Fretless Bass': 35,
        'Slap Bass 1': 36, 'Slap Bass 2': 37, 'Synth Bass 1': 38, 'Synth Bass 2': 39,
        'Violin': 40, 'Viola': 41, 'Cello': 42, 'Contrabass': 43, 'Tremolo Strings': 44,
        'Pizzicato Strings': 45, 'Orchestral Harp': 46, 'Timpani': 47, 'String Ensemble 1': 48,
        'String Ensemble 2': 49, 'Synth Strings 1': 50, 'Synth Strings 2': 51, 'Choir Aahs': 52,
        'Voice Oohs': 53, 'Synth Voice': 54, 'Orchestra Hit': 55, 'Trumpet': 56, 'Trombone': 57,
        'Tuba': 58, 'Muted Trumpet': 59, 'French Horn': 60, 'Brass Section': 61,
        'Synth Brass 1': 62, 'Synth Brass 2': 63, 'Soprano Sax': 64, 'Alto Sax': 65,
        'Tenor Sax': 66, 'Baritone Sax': 67, 'Oboe': 68, 'English Horn': 69, 'Bassoon': 70,
        'Clarinet': 71, 'Piccolo': 72, 'Flute': 73, 'Recorder': 74, 'Pan Flute': 75,
        'Blown Bottle': 76, 'Shakuhachi': 77, 'Whistle': 78, 'Ocarina': 79,
        'Lead 1 (square)': 80, 'Lead 2 (sawtooth)': 81, 'Lead 3 (calliope)': 82,
        'Lead 4 (chiff)': 83, 'Lead 5 (charang)': 84, 'Lead 6 (voice)': 85,
        'Lead 7 (fifths)': 86, 'Lead 8 (bass + lead)': 87, 'Pad 1 (new age)': 88,
        'Pad 2 (warm)': 89, 'Pad 3 (polysynth)': 90, 'Pad 4 (choir)': 91, 'Pad 5 (bowed)': 92,
        'Pad 6 (metallic)': 93, 'Pad 7 (halo)': 94, 'Pad 8 (sweep)': 95, 'FX 1 (rain)': 96,
        'FX 2 (soundtrack)': 97, 'FX 3 (crystal)': 98, 'FX 4 (atmosphere)': 99,
        'FX 5 (brightness)': 100, 'FX 6 (goblins)': 101, 'FX 7 (echoes)': 102,
        'FX 8 (sci-fi)': 103, 'Sitar': 104, 'Banjo': 105, 'Shamisen': 106, 'Koto': 107,
        'Kalimba': 108, 'Bagpipe': 109, 'Fiddle': 110, 'Shanai': 111, 'Tinkle Bell': 112,
        'Agogo': 113, 'Steel Drums': 114, 'Woodblock': 115, 'Taiko Drum': 116,
        'Melodic Tom': 117, 'Synth Drum': 118, 'Reverse Cymbal': 119,
        'Guitar Fret Noise': 120, 'Breath Noise': 121, 'Seashore': 122, 'Bird Tweet': 123,
        'Telephone Ring': 124, 'Helicopter': 125, 'Applause': 126, 'Gunshot': 127, 'Drums': 0
    }
    
    STYLES = ['流行', '民谣', '摇滚', '中国风', '说唱', 'R&B', '舞曲']
    
    BPM_RANGE = (60, 180)
    BARS_RANGE = (4, 32)
