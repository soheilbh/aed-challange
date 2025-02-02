import os
import pandas as pd
import numpy as np
from scipy.io import wavfile
import librosa

def load_metadata(csv_file_path):
    """
    Reads the CSV file into a pandas DataFrame.
    """
    return pd.read_csv(csv_file_path)


def get_audio_file(audio_files_path, filename: str):
    """
    Constructs the full file path for a given audio filename.
    """
    return os.path.join(audio_files_path, filename)


def sample_sounds_by_category(df, categories: dict):
    """
    Given a DataFrame (df) of the ESC-50 metadata and a dict of categories to sounds,
    returns a dict of category -> list of sampled filenames.
    """
    sampled_sounds = {}
    
    for category, sounds in categories.items():
        # Filter df rows matching any sound in this category
        category_df = df[df['category'].isin(sounds)]
        
        # For each specific sound in the category, sample 1 row
        samples = pd.concat([
            category_df[category_df['category'] == sound].sample(n=1)
            for sound in sounds
        ])
        
        # Store filenames for the chosen samples
        sampled_sounds[category] = samples['filename'].tolist()
    
    return sampled_sounds

def load_all_sounds(df, categories, audio_files_path):
    
    sound_data_list = []

    for category, classes in categories.items():
        for sound_class in classes:
            # Filter rows for the specific class
            class_files = df[df['category'] == sound_class]['filename'].tolist()
            
            for filename in class_files:
                file_path = os.path.join(audio_files_path, filename)
                try:
                    # Load the audio file using librosa
                    sound_data, sample_rate = librosa.load(file_path, sr=None)
                    sound_data_list.append((category, filename, sample_rate, sound_data))
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

    return sound_data_list

def load_wave_data(sampled_sounds, audio_files_path):
    """
    Given the sampled_sounds dict and path to audio files,
    reads each WAV file into a list of tuples:
    [(category, filename, sample_rate, sound_data), ...]
    """
    wave_list_data = []
    
    for category in sampled_sounds:
        for filename in sampled_sounds[category]:
            complete_path = get_audio_file(audio_files_path, filename)
            sample_rate, sound_data = wavfile.read(complete_path)
            wave_list_data.append((category, filename, sample_rate, sound_data))
            
    return wave_list_data


def group_data_by_category(wave_list_data):
    """
    Optional utility: Groups wave data by category into a dict.
    """
    category_groups = {}
    for wave in wave_list_data:
        category, filename, sample_rate, sound_data = wave
        if category not in category_groups:
            category_groups[category] = []
        category_groups[category].append((category, filename, sample_rate, sound_data))
    return category_groups
