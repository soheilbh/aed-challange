import numpy as np
import pandas as pd
import librosa
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

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

    # Get the number of rows from any feature as a reference
    reference_rows = list(feature_map.values())[0].shape[0]

    # Loop through feature flags and select features with True flag
    for feature, include in feature_flags.items():
        if include:
            if feature in feature_map:
                feature_array = feature_map[feature]
                
                # Assert that the feature array has the correct number of rows
                assert feature_array.shape[0] == reference_rows, f"Feature {feature} has a mismatched number of rows."

                feature_arrays.append(feature_array)
            else:
                print(f"Warning: {feature} is not available in the loaded data.")

    # Handle the case where no features are selected
    if not feature_arrays:
        raise ValueError("No features were selected. Check the feature selection flags.")

    # Horizontally stack the selected feature arrays
    combined_features = np.hstack(feature_arrays)
    print("Combined feature shape:", combined_features.shape)
    return combined_features


# Function to test different feature combinations with optional normalization and PCA
def test_feature_combinations_svm(combinations_dict, y, use_gridsearch=False, svm_params=None,
                                  n_splits=5, n_pca_components=10, normalize=True, apply_pca=True, 
                                  overfit_threshold=0.1, hyperparameter_grid=None):
    results = []

    # Default SVM parameters if not provided
    if svm_params is None:
        svm_params = {'kernel': 'rbf', 'C': 10, 'gamma': 'scale', 'probability': True, 'random_state': 42}

    # Default hyperparameter grid if not provided
    if hyperparameter_grid is None:
        hyperparameter_grid = {'C': [0.1, 1, 10], 'gamma': ['scale'], 'kernel': ['rbf']}

    # Generate all possible non-empty combinations of features
    all_combinations = [comb for i in range(1, len(combinations_dict) + 1) for comb in itertools.combinations(combinations_dict.keys(), i)]

    for combination in all_combinations:
        # Combine selected features
        selected_features = np.hstack([combinations_dict[feature] for feature in combination])

        # Step 1: Normalize features if enabled
        if normalize:
            scaler = StandardScaler()
            selected_features = scaler.fit_transform(selected_features)

        # Step 2: Apply PCA if enabled and feature dimension is high enough
        if apply_pca and selected_features.shape[1] > n_pca_components:
            pca = PCA(n_components=n_pca_components)
            selected_features = pca.fit_transform(selected_features)

        # Step 3: Train-test split
        X_train, X_test, y_train, y_test = train_test_split(selected_features, y, test_size=0.2, random_state=42, stratify=y)

        # Option to use fixed hyperparameters or GridSearchCV
        if use_gridsearch:
            kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            grid_search = GridSearchCV(SVC(probability=True), hyperparameter_grid, scoring='roc_auc_ovr', cv=kfold, n_jobs=-1, verbose=0)
            grid_search.fit(X_train, y_train)
            best_params = grid_search.best_params_
        else:
            best_params = svm_params  # Use provided or default SVM parameters for testing

        # Step 4: Train and evaluate using K-Fold
        train_accuracies, val_accuracies, auc_scores = [], [], []
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        for train_idx, val_idx in kfold.split(X_train, y_train):
            X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
            y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

            svm = SVC(**best_params)
            svm.fit(X_train_fold, y_train_fold)

            # Training and validation accuracy
            train_acc = accuracy_score(y_train_fold, svm.predict(X_train_fold))
            val_acc = accuracy_score(y_val_fold, svm.predict(X_val_fold))

            # AUC score
            y_val_prob = svm.predict_proba(X_val_fold)
            auc_score = roc_auc_score(y_val_fold, y_val_prob, multi_class='ovr')

            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            auc_scores.append(auc_score)

        # Calculate averages
        avg_train_accuracy = np.mean(train_accuracies)
        avg_val_accuracy = np.mean(val_accuracies)
        avg_auc = np.mean(auc_scores)

        # Overfitting detection
        overfitting = avg_train_accuracy - avg_val_accuracy
        overfitting_status = "⚠️ Overfitting Risk" if avg_train_accuracy == 1.0 or overfitting > overfit_threshold else "✅ No Overfitting"

        # Calculate a balanced score using normalized AUC and accuracy
        combined_score = 0.5 * (avg_auc / 1.0 + avg_val_accuracy / 1.0)

        # Store results
        results.append({
            'combination': combination,
            'average_auc': avg_auc,
            'average_train_accuracy': avg_train_accuracy,
            'average_val_accuracy': avg_val_accuracy,
            'overfitting_status': overfitting_status,
            'combined_score': combined_score,
            'best_params': best_params
        })
        print(f"Tested {combination}: AUC={avg_auc:.4f}, Train Acc={avg_train_accuracy:.4f}, Val Acc={avg_val_accuracy:.4f}, Overfitting={overfitting_status}")

    # Filter non-overfitted combinations and sort by combined score
    non_overfitted_results = [res for res in results if res['overfitting_status'] == "✅ No Overfitting"]
    non_overfitted_results.sort(key=lambda x: x['combined_score'], reverse=True)

    return non_overfitted_results

# Function to test different feature combinations with optional normalization and PCA using Random Forest
def test_feature_combinations_rf(combinations_dict, y, use_gridsearch=False, rf_params=None,
                                 n_splits=5, n_pca_components=10, normalize=True, apply_pca=True, 
                                 overfit_threshold=0.1, hyperparameter_grid=None):
    results = []

    # Default Random Forest parameters if not provided
    if rf_params is None:
        rf_params = {'n_estimators': 100, 'max_depth': None, 'random_state': 42}

    # Default hyperparameter grid if not provided
    if hyperparameter_grid is None:
        hyperparameter_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }

    # Generate all possible non-empty combinations of features
    all_combinations = [comb for i in range(1, len(combinations_dict) + 1) for comb in itertools.combinations(combinations_dict.keys(), i)]

    for combination in all_combinations:
        # Combine selected features
        selected_features = np.hstack([combinations_dict[feature] for feature in combination])

        # Step 1: Normalize features if enabled
        if normalize:
            scaler = StandardScaler()
            selected_features = scaler.fit_transform(selected_features)

        # Step 2: Apply PCA if enabled and feature dimension is high enough
        if apply_pca and selected_features.shape[1] > n_pca_components:
            pca = PCA(n_components=n_pca_components)
            selected_features = pca.fit_transform(selected_features)

        # Step 3: Train-test split
        X_train, X_test, y_train, y_test = train_test_split(selected_features, y, test_size=0.2, random_state=42, stratify=y)

        # Option to use fixed hyperparameters or GridSearchCV
        if use_gridsearch:
            kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            grid_search = GridSearchCV(RandomForestClassifier(), hyperparameter_grid, scoring='roc_auc_ovr', cv=kfold, n_jobs=-1, verbose=0)
            grid_search.fit(X_train, y_train)
            best_params = grid_search.best_params_
        else:
            best_params = rf_params  # Use provided or default Random Forest parameters for testing

        # Step 4: Train and evaluate using K-Fold
        train_accuracies, val_accuracies, auc_scores = [], [], []
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        for train_idx, val_idx in kfold.split(X_train, y_train):
            X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
            y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

            rf = RandomForestClassifier(**best_params)
            rf.fit(X_train_fold, y_train_fold)

            # Training and validation accuracy
            train_acc = accuracy_score(y_train_fold, rf.predict(X_train_fold))
            val_acc = accuracy_score(y_val_fold, rf.predict(X_val_fold))

            # AUC score
            y_val_prob = rf.predict_proba(X_val_fold)
            auc_score = roc_auc_score(y_val_fold, y_val_prob, multi_class='ovr')

            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            auc_scores.append(auc_score)

        # Calculate averages
        avg_train_accuracy = np.mean(train_accuracies)
        avg_val_accuracy = np.mean(val_accuracies)
        avg_auc = np.mean(auc_scores)

        # Overfitting detection
        overfitting = avg_train_accuracy - avg_val_accuracy
        overfitting_status = "⚠️ Overfitting Risk" if avg_train_accuracy == 1.0 or overfitting > overfit_threshold else "✅ No Overfitting"

        # Calculate a balanced score using normalized AUC and accuracy
        combined_score = 0.5 * (avg_auc / 1.0 + avg_val_accuracy / 1.0)

        # Store results
        results.append({
            'combination': combination,
            'average_auc': avg_auc,
            'average_train_accuracy': avg_train_accuracy,
            'average_val_accuracy': avg_val_accuracy,
            'overfitting_status': overfitting_status,
            'combined_score': combined_score,
            'best_params': best_params
        })
        print(f"Tested {combination}: AUC={avg_auc:.4f}, Train Acc={avg_train_accuracy:.4f}, Val Acc={avg_val_accuracy:.4f}, Overfitting={overfitting_status}")

    # Filter non-overfitted combinations and sort by combined score
    non_overfitted_results = [res for res in results if res['overfitting_status'] == "✅ No Overfitting"]
    non_overfitted_results.sort(key=lambda x: x['combined_score'], reverse=True)

    return non_overfitted_results

def svm_kfold_cross_validation(features, labels, selected_features_names, n_splits=5, n_pca_components=10, 
                               normalize=False, apply_pca=False, svm_params=None, overfit_threshold=0.1):
    # Default SVM parameters if not provided
    if svm_params is None:
        svm_params = {'kernel': 'rbf', 'C': 10, 'gamma': 'scale', 'probability': True, 'random_state': 42}
    
    # Track settings
    normalization_status = "Enabled" if normalize else "Disabled"
    pca_status = f"Enabled (n_components={n_pca_components})" if apply_pca else "Disabled"

    print(f"\nSettings:")
    print(f" - Selected Features: {', '.join(selected_features_names)}")
    print(f" - Normalization: {normalization_status}")
    print(f" - PCA: {pca_status}")
    print(f" - SVM Parameters: {svm_params}\n")

    # Initialize K-Fold Cross-Validation
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    train_accuracies = []
    test_accuracies = []
    auc_scores = []
    conf_matrices = []

    for train_idx, test_idx in kfold.split(features, labels):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        # Step 1: Normalize if enabled
        if normalize:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # Step 2: Apply PCA if enabled
        if apply_pca:
            pca = PCA(n_components=n_pca_components)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)

        # Step 3: Train the SVM classifier
        svm = SVC(**svm_params)
        svm.fit(X_train, y_train)

        # Step 4: Evaluate on training set
        y_train_pred = svm.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_accuracies.append(train_accuracy)

        # Step 5: Evaluate on test set
        y_test_pred = svm.predict(X_test)
        y_test_prob = svm.predict_proba(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_accuracies.append(test_accuracy)

        # Compute AUC score
        auc = roc_auc_score(y_test, y_test_prob, multi_class='ovr')
        auc_scores.append(auc)

        # Save confusion matrix for this fold
        conf_matrices.append(confusion_matrix(y_test, y_test_pred))

    # Step 6: Calculate average metrics
    avg_train_accuracy = np.mean(train_accuracies)
    avg_test_accuracy = np.mean(test_accuracies)
    avg_auc = np.mean(auc_scores)

    print(f"\nK-Fold Cross-Validation Summary:")
    print(f"Average Train Accuracy: {avg_train_accuracy:.4f}")
    print(f"Average Test Accuracy: {avg_test_accuracy:.4f}")
    print(f"Average AUC: {avg_auc:.4f}")

    # Overfitting detection
    overfitting_gap = avg_train_accuracy - avg_test_accuracy
    overfitting_status = "⚠️ Overfitting Risk" if avg_train_accuracy == 1.0 or overfitting_gap > overfit_threshold else "✅ No Overfitting"
    print(f"Overfitting Status: {overfitting_status} (Train-Test Gap: {overfitting_gap:.4f})")

    # --- Plotting Cross-Validation Results ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Boxplot of train and test accuracies across folds
    axes[0].boxplot([train_accuracies, test_accuracies], labels=["Train Accuracy", "Test Accuracy"], patch_artist=True,
                    boxprops=dict(facecolor="skyblue", color="black"), medianprops=dict(color="red"))
    axes[0].set_title("Accuracy per Fold")
    axes[0].set_ylabel("Accuracy")

    # Boxplot of AUC scores
    axes[1].boxplot(auc_scores, labels=["AUC"], patch_artist=True,
                    boxprops=dict(facecolor="lightgreen", color="black"), medianprops=dict(color="red"))
    axes[1].set_title("AUC per Fold")
    axes[1].set_ylabel("AUC Score")

    plt.tight_layout()
    plt.show()

    # --- Confusion Matrix on Last Fold ---
    print("\nSVM - Classification Report (Last Fold):")
    print(classification_report(y_test, y_test_pred))

    # Confusion matrix visualization for the last fold
    plt.figure(figsize=(14, 6))
    sns.heatmap(conf_matrices[-1], annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(labels), yticklabels=np.unique(labels))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('SVM - Confusion Matrix (Last Fold)')
    plt.show()

    # Return key metrics for further analysis
    return {
        'avg_train_accuracy': avg_train_accuracy,
        'avg_test_accuracy': avg_test_accuracy,
        'avg_auc': avg_auc,
        'overfitting_status': overfitting_status,
        'overfitting_gap': overfitting_gap
    }

def rf_kfold_cross_validation(features, labels, selected_features_names, n_splits=5, n_pca_components=10, 
                              normalize=False, apply_pca=False, rf_params=None, overfit_threshold=0.1):
    # Default RF parameters if not provided
    if rf_params is None:
        rf_params = {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5, 'random_state': 42}
    
    # Track settings
    normalization_status = "Enabled" if normalize else "Disabled"
    pca_status = f"Enabled (n_components={n_pca_components})" if apply_pca else "Disabled"

    print(f"\nSettings:")
    print(f" - Selected Features: {', '.join(selected_features_names)}")
    print(f" - Normalization: {normalization_status}")
    print(f" - PCA: {pca_status}")
    print(f" - Random Forest Parameters: {rf_params}\n")

    # Initialize K-Fold Cross-Validation
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    train_accuracies = []
    test_accuracies = []
    auc_scores = []
    conf_matrices = []

    for train_idx, test_idx in kfold.split(features, labels):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        # Step 1: Normalize if enabled
        if normalize:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # Step 2: Apply PCA if enabled
        if apply_pca:
            pca = PCA(n_components=n_pca_components)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)

        # Step 3: Train the Random Forest classifier
        rf = RandomForestClassifier(**rf_params)
        rf.fit(X_train, y_train)

        # Step 4: Evaluate on training set
        y_train_pred = rf.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_accuracies.append(train_accuracy)

        # Step 5: Evaluate on test set
        y_test_pred = rf.predict(X_test)
        y_test_prob = rf.predict_proba(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_accuracies.append(test_accuracy)

        # Compute AUC score
        auc = roc_auc_score(y_test, y_test_prob, multi_class='ovr')
        auc_scores.append(auc)

        # Save confusion matrix for this fold
        conf_matrices.append(confusion_matrix(y_test, y_test_pred))

    # Step 6: Calculate average metrics
    avg_train_accuracy = np.mean(train_accuracies)
    avg_test_accuracy = np.mean(test_accuracies)
    avg_auc = np.mean(auc_scores)

    print(f"\nK-Fold Cross-Validation Summary:")
    print(f"Average Train Accuracy: {avg_train_accuracy:.4f}")
    print(f"Average Test Accuracy: {avg_test_accuracy:.4f}")
    print(f"Average AUC: {avg_auc:.4f}")

    # Overfitting detection
    overfitting_gap = avg_train_accuracy - avg_test_accuracy
    overfitting_status = "⚠️ Overfitting Risk" if avg_train_accuracy == 1.0 or overfitting_gap > overfit_threshold else "✅ No Overfitting"
    print(f"Overfitting Status: {overfitting_status} (Train-Test Gap: {overfitting_gap:.4f})")

    # --- Plotting Cross-Validation Results ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Boxplot of train and test accuracies across folds
    axes[0].boxplot([train_accuracies, test_accuracies], labels=["Train Accuracy", "Test Accuracy"], patch_artist=True,
                    boxprops=dict(facecolor="skyblue", color="black"), medianprops=dict(color="red"))
    axes[0].set_title("Accuracy per Fold")
    axes[0].set_ylabel("Accuracy")

    # Boxplot of AUC scores
    axes[1].boxplot(auc_scores, labels=["AUC"], patch_artist=True,
                    boxprops=dict(facecolor="lightgreen", color="black"), medianprops=dict(color="red"))
    axes[1].set_title("AUC per Fold")
    axes[1].set_ylabel("AUC Score")

    plt.tight_layout()
    plt.show()

    # --- Confusion Matrix on Last Fold ---
    print("\nRandom Forest - Classification Report (Last Fold):")
    print(classification_report(y_test, y_test_pred))

    # Confusion matrix visualization for the last fold
    plt.figure(figsize=(14, 6))
    sns.heatmap(conf_matrices[-1], annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(labels), yticklabels=np.unique(labels))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Random Forest - Confusion Matrix (Last Fold)')
    plt.show()

    # Return key metrics for further analysis
    return {
        'avg_train_accuracy': avg_train_accuracy,
        'avg_test_accuracy': avg_test_accuracy,
        'avg_auc': avg_auc,
        'overfitting_status': overfitting_status,
        'overfitting_gap': overfitting_gap
    }

def knn_kfold_cross_validation(features, labels, selected_features_names, n_splits=5, n_pca_components=10, 
                               normalize=False, apply_pca=False, knn_params=None, overfit_threshold=0.1):
    # Default KNN parameters if not provided
    if knn_params is None:
        knn_params = {'n_neighbors': 5, 'weights': 'uniform', 'metric': 'euclidean'}
    
    # Track settings
    normalization_status = "Enabled" if normalize else "Disabled"
    pca_status = f"Enabled (n_components={n_pca_components})" if apply_pca else "Disabled"

    print(f"\nSettings:")
    print(f" - Selected Features: {', '.join(selected_features_names)}")
    print(f" - Normalization: {normalization_status}")
    print(f" - PCA: {pca_status}")
    print(f" - KNN Parameters: {knn_params}\n")

    # Initialize K-Fold Cross-Validation
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    train_accuracies = []
    test_accuracies = []
    auc_scores = []
    conf_matrices = []

    for train_idx, test_idx in kfold.split(features, labels):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        # Step 1: Normalize if enabled
        if normalize:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # Step 2: Apply PCA if enabled
        if apply_pca:
            pca = PCA(n_components=n_pca_components)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)

        # Step 3: Train the KNN classifier
        knn = KNeighborsClassifier(**knn_params)
        knn.fit(X_train, y_train)

        # Step 4: Evaluate on training set
        y_train_pred = knn.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_accuracies.append(train_accuracy)

        # Step 5: Evaluate on test set
        y_test_pred = knn.predict(X_test)
        y_test_prob = knn.predict_proba(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_accuracies.append(test_accuracy)

        # Compute AUC score
        auc = roc_auc_score(y_test, y_test_prob, multi_class='ovr')
        auc_scores.append(auc)

        # Save confusion matrix for this fold
        conf_matrices.append(confusion_matrix(y_test, y_test_pred))

    # Step 6: Calculate average metrics
    avg_train_accuracy = np.mean(train_accuracies)
    avg_test_accuracy = np.mean(test_accuracies)
    avg_auc = np.mean(auc_scores)

    print(f"\nK-Fold Cross-Validation Summary:")
    print(f"Average Train Accuracy: {avg_train_accuracy:.4f}")
    print(f"Average Test Accuracy: {avg_test_accuracy:.4f}")
    print(f"Average AUC: {avg_auc:.4f}")

    # Overfitting detection
    overfitting_gap = avg_train_accuracy - avg_test_accuracy
    overfitting_status = "⚠️ Overfitting Risk" if avg_train_accuracy == 1.0 or overfitting_gap > overfit_threshold else "✅ No Overfitting"
    print(f"Overfitting Status: {overfitting_status} (Train-Test Gap: {overfitting_gap:.4f})")

    # --- Plotting Cross-Validation Results ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Boxplot of train and test accuracies across folds
    axes[0].boxplot([train_accuracies, test_accuracies], labels=["Train Accuracy", "Test Accuracy"], patch_artist=True,
                    boxprops=dict(facecolor="skyblue", color="black"), medianprops=dict(color="red"))
    axes[0].set_title("Accuracy per Fold")
    axes[0].set_ylabel("Accuracy")

    # Boxplot of AUC scores
    axes[1].boxplot(auc_scores, labels=["AUC"], patch_artist=True,
                    boxprops=dict(facecolor="lightgreen", color="black"), medianprops=dict(color="red"))
    axes[1].set_title("AUC per Fold")
    axes[1].set_ylabel("AUC Score")

    plt.tight_layout()
    plt.show()

    # --- Confusion Matrix on Last Fold ---
    print("\nKNN - Classification Report (Last Fold):")
    print(classification_report(y_test, y_test_pred))

    # Confusion matrix visualization for the last fold
    plt.figure(figsize=(14, 6))
    sns.heatmap(conf_matrices[-1], annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(labels), yticklabels=np.unique(labels))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('KNN - Confusion Matrix (Last Fold)')
    plt.show()

    # Return key metrics for further analysis
    return {
        'avg_train_accuracy': avg_train_accuracy,
        'avg_test_accuracy': avg_test_accuracy,
        'avg_auc': avg_auc,
        'overfitting_status': overfitting_status,
        'overfitting_gap': overfitting_gap
    }

def get_selected_features(feature_combinations, feature_selection):
    selected_features_combination = [feat for feat, selected in feature_selection.items() if selected]
    selected_features = np.hstack([feature_combinations[feat] for feat in selected_features_combination])
    return selected_features, selected_features_combination

def load_and_split_features(loaded_data, feature_selection, test_size=0.2, random_state=42):
    # Extract selected features based on the feature_selection dictionary
    combined_features = combine_features_with_flags(loaded_data, feature_selection)
    selected_features_names = [feature for feature, include in feature_selection.items() if include]

    # Display the shapes of each feature type
    print("Feature shapes and selection status:")
    for feature, include in feature_selection.items():
        shape = loaded_data[feature].shape
        status = "✅" if include else "❌"
        print(f"{feature.capitalize()} shape: {shape} - Status: {status}")
    
    print("\nCombined features shape:", combined_features.shape)
    
    # Extract labels
    y = np.array(loaded_data['keys'])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        combined_features, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return combined_features, X_train, X_test, y_train, y_test, selected_features_names



def preprocess_features(X_train, X_test, normalize=True, apply_pca=False, n_pca_components=0.9, verbose=True):
    # Step 1: Normalize the features (if enabled)
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        if verbose:
            print("✅ Normalization applied.")
    else:
        if verbose:
            print("❌ Normalization skipped.")

    # Step 2: Apply PCA (if enabled)
    if apply_pca:
        pca = PCA(n_components=n_pca_components)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

        # Dynamically retrieve the number of components selected
        n_components_selected = pca.n_components_
        explained_variance = np.sum(pca.explained_variance_ratio_)
        
        if verbose:
            print(f"✅ PCA applied. Selected {n_components_selected} components (explained variance: {explained_variance:.2f}).")
            print(f"Reduced feature shape after PCA: {X_train.shape[1]}")
    else:
        if verbose:
            print("❌ PCA skipped.")

    return X_train, X_test


def grid_search_hyperparameter_tuning(model, param_grid, X_train, y_train, cv=5, scoring='roc_auc_ovr', n_jobs=-1, verbose=1):
    try:
        # Initialize GridSearchCV
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=cv, n_jobs=n_jobs, verbose=verbose)

        # Fit the model to the training data
        grid_search.fit(X_train, y_train)

        # Ensure the objects exist
        best_params = grid_search.best_params_ if grid_search.best_params_ else None
        best_score = grid_search.best_score_ if grid_search.best_score_ else 0.0

        print("\nBest Parameters:", best_params)
        print(f"Best {scoring} Score: {best_score:.4f}")

        # Return all three values
        return grid_search, best_params, best_score

    except Exception as e:
        print(f"Grid search failed with error: {str(e)}")
        return None, None, None

def kfold_cross_validation(features, labels, selected_features_names, model, model_params=None, n_splits=5, 
                           preprocess_params=None, overfit_threshold=0.1):
     # Track settings
    print(f"\nSettings:")
    print(f" - Selected Features: {', '.join(selected_features_names)}")
    print(f" - Model: {model.__class__.__name__}")
    print(f" - Model Parameters: {model_params}\n")
    print(f" - Preprocessing: {preprocess_params}\n")

    # Initialize K-Fold Cross-Validation
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    train_accuracies = []
    test_accuracies = []
    auc_scores = [] if hasattr(model, "predict_proba") else None  # Initialize only if applicable
    conf_matrices = []

    # Loop through each fold
    for fold, (train_idx, test_idx) in enumerate(kfold.split(features, labels)):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        # Display class distribution in each fold correctly using Pandas
        train_distribution = pd.Series(y_train).value_counts().reindex(np.unique(labels), fill_value=0)
        test_distribution = pd.Series(y_test).value_counts().reindex(np.unique(labels), fill_value=0)
        
        # Step 1: Preprocess the features (normalize, apply PCA, etc.)
        X_train, X_test = preprocess_features(
            X_train, X_test, 
            normalize=preprocess_params.get('normalize', False), 
            apply_pca=preprocess_params.get('apply_pca', False), 
            n_pca_components=preprocess_params.get('n_pca_components', 0.9),
            verbose=False  # Disable repetitive messages
        )

        # Display the correct feature dimension after preprocessing
        print(f"\n[Fold {fold + 1}]")
        print(f"   - Training set size: {len(y_train)}, Test set size: {len(y_test)}")
        print(f"   - Feature dimension after preprocessing: {X_train.shape[1]}")
        
        # Step 2: Initialize and train the model
        classifier = model.set_params(**model_params)  # Pass custom parameters
        classifier.fit(X_train, y_train)

        # Step 3: Evaluate on training set
        y_train_pred = classifier.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_accuracies.append(train_accuracy)

        # Step 4: Evaluate on test set
        y_test_pred = classifier.predict(X_test)
        y_test_prob = classifier.predict_proba(X_test) if hasattr(classifier, "predict_proba") else None
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_accuracies.append(test_accuracy)

        # Compute AUC score if applicable
        auc = None
        if auc_scores is not None and y_test_prob is not None:
            auc = roc_auc_score(y_test, y_test_prob, multi_class='ovr')
            auc_scores.append(auc)

        # Save confusion matrix for this fold
        conf_matrices.append(confusion_matrix(y_test, y_test_pred))

        # Display fold-wise metrics
        print(f"   - Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
        if auc is not None:
            print(f"   - AUC: {auc:.4f}")

    # Calculate average metrics
    avg_train_accuracy = np.mean(train_accuracies)
    avg_test_accuracy = np.mean(test_accuracies)
    avg_auc = np.mean(auc_scores) if auc_scores else None

    print(f"\nK-Fold Cross-Validation Summary:")
    print(f"Average Train Accuracy: {avg_train_accuracy:.4f}")
    print(f"Average Test Accuracy: {avg_test_accuracy:.4f}")
    if avg_auc is not None:
        print(f"Average AUC: {avg_auc:.4f}")

    # Overfitting detection
    overfitting_gap = avg_train_accuracy - avg_test_accuracy
    overfitting_status = "⚠️ Overfitting Risk" if avg_train_accuracy == 1.0 or overfitting_gap > overfit_threshold else "✅ No Overfitting"
    print(f"Overfitting Status: {overfitting_status} (Train-Test Gap: {overfitting_gap:.4f})")

    # --- Plotting Cross-Validation Results ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Boxplot of train and test accuracies across folds
    axes[0].boxplot([train_accuracies, test_accuracies], labels=["Train Accuracy", "Test Accuracy"], patch_artist=True,
                    boxprops=dict(facecolor="skyblue", color="black"), medianprops=dict(color="red"))
    axes[0].set_title("Accuracy per Fold")
    axes[0].set_ylabel("Accuracy")

    # Boxplot of AUC scores (if available)
    if auc_scores:
        axes[1].boxplot(auc_scores, labels=["AUC"], patch_artist=True,
                        boxprops=dict(facecolor="lightgreen", color="black"), medianprops=dict(color="red"))
        axes[1].set_title("AUC per Fold")
        axes[1].set_ylabel("AUC Score")

    plt.tight_layout()
    plt.show()

    # --- Confusion Matrix on Last Fold ---
    print("\nClassification Report (Last Fold):")
    print(classification_report(y_test, y_test_pred))

    # Confusion matrix visualization for the last fold
    plt.figure(figsize=(14, 6))
    sns.heatmap(conf_matrices[-1], annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(labels), yticklabels=np.unique(labels))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{model.__class__.__name__} - Confusion Matrix (Last Fold)')
    plt.show()

    # Return key metrics for further analysis
    return {
        'avg_train_accuracy': avg_train_accuracy,
        'avg_test_accuracy': avg_test_accuracy,
        'avg_auc': avg_auc,
        'overfitting_status': overfitting_status,
        'overfitting_gap': overfitting_gap
    }