import numpy as np
import librosa

def extract_mfcc(sound_data, sample_rate, n_mfcc=13, n_fft=2048, hop_length=512):
    """
    Extract MFCC features from audio data.
    Returns an array of shape (n_mfcc, T) where T is the number of frames.
    """
    # Convert to float32 in case the array is int16
    sound_data_float = sound_data.astype(np.float32)
    
    # Compute MFCC
    mfcc = librosa.feature.mfcc(
        y=sound_data_float,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length
    )
    return mfcc


def extract_binned_spectrogram_hist(sound_data, sample_rate, 
                                    n_fft=2048, hop_length=512, 
                                    n_bands=8, n_magnitude_bins=10):
    """
    Calculate a spectrogram, split frequency into n_bands,
    and compute a histogram of magnitudes for each band.
    Returns a 1D feature vector.
    """
    import librosa  # Ensure librosa is available here
    sound_data_float = sound_data.astype(np.float32)
    
    # Compute STFT -> shape: (1 + n_fft/2, num_frames)
    stft = librosa.stft(sound_data_float, n_fft=n_fft, hop_length=hop_length)
    spectrogram = np.abs(stft)
    freq_bins = spectrogram.shape[0]
    
    # Divide freq_bins into n_bands
    band_size = freq_bins // n_bands
    hist_features = []
    
    for b in range(n_bands):
        start_idx = b * band_size
        # last band might be bigger if freq_bins not divisible by n_bands
        end_idx = freq_bins if b == n_bands - 1 else (b + 1) * band_size
        
        band_data = spectrogram[start_idx:end_idx, :]
        band_magnitudes = band_data.flatten()
        
        # Compute histogram
        # We add a small epsilon in case band_magnitudes.max() is zero
        hist, _ = np.histogram(
            band_magnitudes, bins=n_magnitude_bins, 
            range=(0, band_magnitudes.max() + 1e-9)
        )
        # Normalize histogram
        hist = hist / (np.sum(hist) + 1e-9)
        
        hist_features.extend(hist.tolist())
    
    return np.array(hist_features)


def compute_features_for_wave_list(wave_list_data):
    """
    Loop over wave_list_data ([(category, filename, sr, sound_data), ...])
    and compute features (e.g., MFCC and binned hist).
    
    Returns:
        keys_list: list of unique IDs (like "category_filename")
        mfcc_list: list of MFCC feature vectors (mean+std for each file)
        hist_list: list of binned spectrogram hist vectors
    """
    keys_list = []
    mfcc_list = []
    hist_list = []
    spectral_list = []
    
    for category, filename, sample_rate, sound_data in wave_list_data:
        audio_key = f"{category}_{filename}"
        keys_list.append(audio_key)
        
        # 1) Extract MFCC, then reduce to mean & std
        mfcc_matrix = extract_mfcc(sound_data, sample_rate)
        mfcc_mean = np.mean(mfcc_matrix, axis=1)
        mfcc_std = np.std(mfcc_matrix, axis=1)
        mfcc_feature_vector = np.concatenate([mfcc_mean, mfcc_std])
        
        # 2) Extract binned spectrogram histogram
        hist_feature_vector = extract_binned_spectrogram_hist(sound_data, sample_rate)

        # 3) Extract spectral features (centroid, contrast, pitch)
        spectral_feature_vector = extract_spectral_features(sound_data, sample_rate)
        
        # Store results
        mfcc_list.append(mfcc_feature_vector)
        hist_list.append(hist_feature_vector)
        spectral_list.append(spectral_feature_vector)
    
    return keys_list, mfcc_list, hist_list, spectral_list


def save_features_to_npz(keys_list, mfcc_list, hist_list, spectral_list, out_file="features.npz"):
    """
    Converts the lists of feature vectors into numpy arrays and saves them to npz.
    """
    mfcc_array = np.vstack(mfcc_list)  # shape: (num_files, vector_dim)
    hist_array = np.vstack(hist_list)  # shape: (num_files, vector_dim)
    spectral_array = np.vstack(spectral_list)
    
    np.savez(
        out_file,
        keys=keys_list,
        mfcc=mfcc_array,
        hist=hist_array,
        spectral=spectral_array
    )
    
    print(f"Features saved to {out_file}")

def extract_spectral_features(sound_data, sample_rate, n_fft=2048, hop_length=512):
    """
    Extracts spectral features: spectral centroid, spectral contrast, and pitch.
    Returns a feature vector combining these features.
    """
    sound_data_float = sound_data.astype(np.float32)

    # Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=sound_data_float, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)
    spectral_centroid_mean = np.mean(spectral_centroid)
    spectral_centroid_std = np.std(spectral_centroid)

    # Spectral Contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=sound_data_float, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)
    spectral_contrast_mean = np.mean(spectral_contrast, axis=1)
    spectral_contrast_std = np.std(spectral_contrast, axis=1)

    # Pitch (Fundamental Frequency)
    pitches, magnitudes = librosa.piptrack(y=sound_data_float, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)
    pitch_mean = np.mean(pitches[pitches > 0]) if pitches[pitches > 0].size > 0 else 0
    pitch_std = np.std(pitches[pitches > 0]) if pitches[pitches > 0].size > 0 else 0

    # Combine features into one vector
    feature_vector = np.concatenate(
        [
            [spectral_centroid_mean, spectral_centroid_std],  # Spectral Centroid
            spectral_contrast_mean, spectral_contrast_std,    # Spectral Contrast
            [pitch_mean, pitch_std]                          # Pitch
        ]
    )

    return feature_vector
