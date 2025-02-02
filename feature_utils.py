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

def extract_spectral_centroid(sound_data, sample_rate, n_fft=2048, hop_length=512):
    """ Extracts the spectral centroid feature (Mean & Std) """
    sound_data_float = sound_data.astype(np.float32)
    centroid = librosa.feature.spectral_centroid(y=sound_data_float, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)
    return np.array([np.mean(centroid), np.std(centroid)])

def extract_spectral_contrast(sound_data, sample_rate, n_fft=2048, hop_length=512):
    """ Extracts the spectral contrast feature (Mean & Std for each band) """
    sound_data_float = sound_data.astype(np.float32)
    contrast = librosa.feature.spectral_contrast(y=sound_data_float, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)
    contrast_mean = np.mean(contrast, axis=1)
    contrast_std = np.std(contrast, axis=1)
    return np.concatenate([contrast_mean, contrast_std])

def extract_pitch(sound_data, sample_rate, n_fft=2048, hop_length=512):
    """ Extracts pitch feature (Mean & Std) """
    sound_data_float = sound_data.astype(np.float32)
    pitches, _ = librosa.piptrack(y=sound_data_float, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)
    pitch_values = pitches[pitches > 0]
    pitch_mean = np.mean(pitch_values) if pitch_values.size > 0 else 0
    pitch_std = np.std(pitch_values) if pitch_values.size > 0 else 0
    return np.array([pitch_mean, pitch_std])

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

def compute_combined_features_for_wave_list(wave_list_data):
    keys_list = []
    feature_list = []

    for category, filename, sample_rate, sound_data in wave_list_data:
        # Extract class number directly from filename (before .wav)
        class_number = int(filename.split('-')[-1].replace('.wav', ''))
        keys_list.append(class_number)

        # Step 1: Extract MFCCs and delta MFCCs
        mfcc_matrix = librosa.feature.mfcc(y=sound_data, sr=sample_rate, n_mfcc=13)
        delta_mfcc = librosa.feature.delta(mfcc_matrix)

        # Compute mean and std of MFCCs and delta MFCCs
        mfcc_mean = np.mean(mfcc_matrix, axis=1)
        delta_mfcc_mean = np.mean(delta_mfcc, axis=1)
        mfcc_feature_vector = np.concatenate([mfcc_mean, delta_mfcc_mean])

        # Step 2: Extract histogram features
        hist_feature_vector = extract_binned_spectrogram_hist(sound_data, sample_rate)

        # Step 3: Extract spectral features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=sound_data, sr=sample_rate))
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=sound_data, sr=sample_rate))
        pitch_features = np.mean(librosa.feature.zero_crossing_rate(y=sound_data))

        # Step 4: Combine all features into one vector
        feature_vector = np.concatenate([
            mfcc_feature_vector,
            hist_feature_vector,
            [spectral_centroid],
            [spectral_contrast],
            [pitch_features]
        ])

        feature_list.append(feature_vector)

    return keys_list, feature_list

def compute_features_for_wave_list(wave_list_data):
    keys_list = []
    mfcc_list = []
    hist_list = []
    spectral_list = []

    for category, filename, sample_rate, sound_data in wave_list_data:
        # Extract class number directly from filename (before .wav)
        class_number = int(filename.split('-')[-1].replace('.wav', ''))
        keys_list.append(class_number)

        # Step 1: Extract MFCCs and delta MFCCs
        mfcc_matrix = librosa.feature.mfcc(y=sound_data, sr=sample_rate, n_mfcc=13)
        delta_mfcc = librosa.feature.delta(mfcc_matrix)

        # Compute mean and std of MFCCs and delta MFCCs
        mfcc_mean = np.mean(mfcc_matrix, axis=1)
        delta_mfcc_mean = np.mean(delta_mfcc, axis=1)
        mfcc_list.append(np.concatenate([mfcc_mean, delta_mfcc_mean]))

        # Step 2: Extract histogram features
        hist_list.append(extract_binned_spectrogram_hist(sound_data, sample_rate))

        # Step 3: Extract spectral features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=sound_data, sr=sample_rate))
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=sound_data, sr=sample_rate))
        pitch_features = np.mean(librosa.feature.zero_crossing_rate(y=sound_data))
        spectral_list.append(np.array([spectral_centroid, spectral_contrast, pitch_features]))

    return keys_list, mfcc_list, hist_list, spectral_list

def save_features_to_npz(keys_list, feature_list, out_file="features.npz"):
    """
    Saves feature vectors to an NPZ file.
    """
    feature_array = np.vstack(feature_list)  
    np.savez(out_file, keys=keys_list, features=feature_array)
    print(f"Features saved to {out_file}")

def save_multiple_features_to_npz(keys_list, mfcc_list, hist_list, spectral_list, out_file="features_multiple.npz"):

    np.savez(out_file, 
             keys=keys_list, 
             mfcc=np.vstack(mfcc_list), 
             hist=np.vstack(hist_list), 
             spectral=np.vstack(spectral_list))
    print(f"Multiple features saved to {out_file}")