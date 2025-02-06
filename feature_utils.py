import numpy as np
import librosa
import itertools
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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

def extract_spectral_contrast(sound_data, sample_rate, n_fft=2048, hop_length=512,n_bands=7):
    """ Extracts the spectral contrast feature (Mean & Std for each band) """
    sound_data_float = sound_data.astype(np.float32)
    contrast = librosa.feature.spectral_contrast(y=sound_data_float, sr=sample_rate, n_fft=n_fft, hop_length=hop_length,n_bands=n_bands)
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

def extract_zcr(sound_data, sample_rate, frame_length=2048, hop_length=512):

    zcr = librosa.feature.zero_crossing_rate(y=sound_data, frame_length=frame_length, hop_length=hop_length)
    mean_zcr = np.mean(zcr)
    std_zcr = np.std(zcr)
    return np.array([mean_zcr, std_zcr])

def extract_amplitude_envelope_features(sound_data, sample_rate, frame_length=2048, hop_length=512):

    rms_energy = librosa.feature.rms(y=sound_data, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Energy statistics
    mean_energy = np.mean(rms_energy)
    max_energy = np.max(rms_energy)
    energy_variance = np.var(rms_energy)
    
    # Attack and decay rates (differences in rising/falling energy)
    attack_rate = np.mean(np.diff(rms_energy[rms_energy > 0]))
    decay_rate = np.mean(np.diff(rms_energy[rms_energy > 0]) * -1)
    
    return np.array([mean_energy, max_energy, energy_variance, attack_rate, decay_rate])

def compute_harmonic_to_noise_ratio(sound_data, sample_rate, n_fft=2048, hop_length=512):

    # Harmonic-percussive separation
    harmonic, percussive = librosa.effects.hpss(sound_data)
    
    # Compute energy of harmonic and percussive parts
    harmonic_energy = np.sum(harmonic ** 2)
    percussive_energy = np.sum(percussive ** 2)
    
    # Compute HNR
    hnr = harmonic_energy / (percussive_energy + 1e-9)  # Small value to avoid division by zero
    return np.array([hnr])

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

        # Step 3: Extract spectral features using defined functions
        spectral_centroid = extract_spectral_centroid(sound_data, sample_rate)
        spectral_contrast = extract_spectral_contrast(sound_data, sample_rate, n_bands=7)
        pitch_features = extract_pitch(sound_data, sample_rate)

        # Step 4: Extract ZCR, Amplitude Envelope, and HNR features
        zcr_features = extract_zcr(sound_data, sample_rate)  # [mean ZCR, std ZCR]
        envelope_features = extract_amplitude_envelope_features(sound_data, sample_rate)  # [mean, max, variance, attack, decay]
        hnr_features = compute_harmonic_to_noise_ratio(sound_data, sample_rate)  # [HNR]

        # Step 5: Combine all features into one vector
        feature_vector = np.concatenate([
            mfcc_feature_vector,
            hist_feature_vector,
            spectral_centroid,     # Spectral centroid (mean, std)
            spectral_contrast,     # Spectral contrast features
            pitch_features,        # Pitch features (mean, std)
            zcr_features,          # Zero-Crossing Rate features
            envelope_features,     # Amplitude envelope features
            hnr_features           # Harmonic-to-Noise Ratio
        ])

        feature_list.append(feature_vector)

    return keys_list, feature_list



def compute_features_for_wave_list(wave_list_data):
    keys_list = []
    mfcc_list = []
    hist_list = []
    spectral_list = []
    zcr_list = []
    envelope_list = []
    hnr_list = []

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

        # Step 3: Extract spectral features using defined functions
        spectral_centroid = extract_spectral_centroid(sound_data, sample_rate)  # Shape: (2,)
        spectral_contrast = extract_spectral_contrast(sound_data, sample_rate, n_bands=7)  # Shape: (2 * n_bands,)
        pitch_features = extract_pitch(sound_data, sample_rate)  # Shape: (2,)

        # Concatenate all spectral features into a single vector
        spectral_feature_vector = np.concatenate([
            spectral_centroid,
            spectral_contrast,
            pitch_features
        ])

        spectral_list.append(spectral_feature_vector)

        # Step 4: Zero-Crossing Rate (ZCR)
        zcr_features = extract_zcr(sound_data, sample_rate)
        zcr_list.append(zcr_features)

        # Step 5: Amplitude Envelope (Energy Features)
        envelope_features = extract_amplitude_envelope_features(sound_data, sample_rate)
        envelope_list.append(envelope_features)

        # Step 6: Harmonic-to-Noise Ratio (HNR)
        hnr_features = compute_harmonic_to_noise_ratio(sound_data, sample_rate)
        hnr_list.append(hnr_features)

    return keys_list, mfcc_list, hist_list, spectral_list, zcr_list, envelope_list, hnr_list


def save_features_to_npz(keys_list, feature_list, out_file="features.npz"):
    """
    Saves feature vectors to an NPZ file.
    """
    feature_array = np.vstack(feature_list)  
    np.savez(out_file, keys=keys_list, features=feature_array)
    print(f"Features saved to {out_file}")

def save_multiple_features_to_npz(keys_list, mfcc_list, hist_list, spectral_list, zcr_list, envelope_list, hnr_list, out_file="features_multiple.npz"):

    np.savez(out_file, 
             keys=keys_list, 
             mfcc=np.vstack(mfcc_list), 
             hist=np.vstack(hist_list), 
             spectral=np.vstack(spectral_list), 
             zcr=np.vstack(zcr_list), 
             envelope=np.vstack(envelope_list), 
             hnr=np.vstack(hnr_list))
    
    print(f"Multiple features saved to {out_file}")

def combine_features_with_flags(loaded_data, feature_flags):
    
    feature_arrays = []

    # Map feature names to the arrays in loaded_data
    feature_map = {
        'mfcc': loaded_data['mfcc'],
        'hist': loaded_data['hist'],
        'spectral': loaded_data['spectral'],
        'zcr': loaded_data['zcr'],
        'envelope': loaded_data['envelope'],
        'hnr': loaded_data['hnr']
    }

    # Loop through feature flags and select features with True flag
    for feature, include in feature_flags.items():
        if include:
            if feature in feature_map:
                feature_arrays.append(feature_map[feature])
            else:
                print(f"Warning: {feature} is not available in the loaded data.")

    # Horizontally stack the selected feature arrays
    combined_features = np.hstack(feature_arrays)
    return combined_features

# Function to test different feature combinations with optional normalization and PCA
def test_feature_combinations_svm(combinations_dict, y, n_splits=5, n_pca_components=10, normalize=True, apply_pca=True):
    results = []
    
    # Generate all possible non-empty combinations of features
    all_combinations = []
    feature_keys = list(combinations_dict.keys())
    for i in range(1, len(feature_keys) + 1):
        all_combinations.extend(itertools.combinations(feature_keys, i))
    
    for combination in all_combinations:
        # Combine selected features
        selected_features = np.hstack([combinations_dict[feature] for feature in combination])
        
        # Step 1: Normalize features if enabled
        if normalize:
            scaler = StandardScaler()
            selected_features = scaler.fit_transform(selected_features)
            print(f"Applied normalization for combination: {combination}")
        
        # Step 2: Apply PCA if enabled and feature dimension is high enough
        if apply_pca and selected_features.shape[1] > n_pca_components:
            pca = PCA(n_components=n_pca_components)
            selected_features = pca.fit_transform(selected_features)
            print(f"Applied PCA: Reduced to {n_pca_components} components for combination {combination}")
        
        # Step 3: Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            selected_features, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train and evaluate using Stratified K-Fold Cross-Validation
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        auc_scores = []
        accuracy_scores = []

        for train_idx, val_idx in kfold.split(X_train, y_train):
            X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
            y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

            # Train SVM
            svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
            svm.fit(X_train_fold, y_train_fold)

            # Make predictions
            y_val_pred = svm.predict(X_val_fold)
            y_val_prob = svm.predict_proba(X_val_fold)

            # Calculate AUC and accuracy
            auc_score = roc_auc_score(y_val_fold, y_val_prob, multi_class='ovr')
            accuracy = accuracy_score(y_val_fold, y_val_pred)

            auc_scores.append(auc_score)
            accuracy_scores.append(accuracy)

        # Average AUC and accuracy over folds
        avg_auc = np.mean(auc_scores)
        avg_accuracy = np.mean(accuracy_scores)
        
        # Store the results
        results.append({
            'combination': combination,
            'average_auc': avg_auc,
            'average_accuracy': avg_accuracy
        })
        print(f"Tested combination: {combination}, AUC = {avg_auc:.4f}, Accuracy = {avg_accuracy:.4f}")

    # Sort results by AUC in descending order
    results.sort(key=lambda x: x['average_auc'], reverse=True)

    return results