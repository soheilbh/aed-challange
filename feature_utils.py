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
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, adjusted_rand_score, confusion_matrix
from scipy.stats import mode

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


def compute_features_for_wave_list(wave_list_data):
    keys_list = []
    mfcc_list = []
    delta_mfcc_list = []
    hist_list = []
    spectral_centroid_list = []
    spectral_contrast_list = []
    pitch_features_list = []
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
        mfcc_list.append(mfcc_mean)
        delta_mfcc_list.append(delta_mfcc_mean)

        # Step 2: Extract histogram features
        hist_list.append(extract_binned_spectrogram_hist(sound_data, sample_rate))

        # Step 3: Extract spectral features using defined functions
        spectral_centroid = extract_spectral_centroid(sound_data, sample_rate)  # Shape: (2,)
        spectral_contrast = extract_spectral_contrast(sound_data, sample_rate, n_bands=7)  # Shape: (2 * n_bands,)
        pitch_features = extract_pitch(sound_data, sample_rate)  # Shape: (2,)

        spectral_centroid_list.append(spectral_centroid)
        spectral_contrast_list.append(spectral_contrast)
        pitch_features_list.append(pitch_features)

        # Step 4: Zero-Crossing Rate (ZCR)
        zcr_features = extract_zcr(sound_data, sample_rate)
        zcr_list.append(zcr_features)

        # Step 5: Amplitude Envelope (Energy Features)
        envelope_features = extract_amplitude_envelope_features(sound_data, sample_rate)
        envelope_list.append(envelope_features)

        # Step 6: Harmonic-to-Noise Ratio (HNR)
        hnr_features = compute_harmonic_to_noise_ratio(sound_data, sample_rate)
        hnr_list.append(hnr_features)

    return keys_list, mfcc_list, delta_mfcc_list, hist_list, spectral_centroid_list, spectral_contrast_list, pitch_features_list, zcr_list, envelope_list, hnr_list


def save_features_to_npz(keys_list, feature_list, out_file="features.npz"):
    """
    Saves feature vectors to an NPZ file.
    """
    feature_array = np.vstack(feature_list)  
    np.savez(out_file, keys=keys_list, features=feature_array)
    print(f"Features saved to {out_file}")

def save_multiple_features_to_npz(keys_list, mfcc_list, delta_mfcc_list, hist_list, spectral_centroid_list, spectral_contrast_list, pitch_features_list, zcr_list, envelope_list, hnr_list, out_file="features_multiple.npz"):

    np.savez(out_file, 
             keys=keys_list, 
             mfcc=np.vstack(mfcc_list), 
             delta_mfcc=np.vstack(delta_mfcc_list),
             hist=np.vstack(hist_list),  
             spectral_centroid=np.vstack(spectral_centroid_list),
             spectral_contrast=np.vstack(spectral_contrast_list),
             pitch_features=np.vstack(pitch_features_list),
             zcr=np.vstack(zcr_list), 
             envelope=np.vstack(envelope_list), 
             hnr=np.vstack(hnr_list))
    
    print(f"Multiple features saved to {out_file}")

def combine_features_with_flags(loaded_data, feature_flags):
    feature_arrays = []

    # Map feature names to the arrays in loaded_data
    feature_map = {
        'mfcc': loaded_data['mfcc'],
        'delta_mfcc': loaded_data['delta_mfcc'],
        'hist': loaded_data['hist'],
        'spectral_centroid': loaded_data['spectral_centroid'],
        'spectral_contrast': loaded_data['spectral_contrast'],
        'pitch_features': loaded_data['pitch_features'],
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


def test_feature_combinations(
    combinations_dict,
    y,
    model,
    model_name,
    model_params=None,
    n_splits=5,
    normalize=True,
    apply_pca=True,
    n_pca_components=0.9,
    overfit_threshold=0.1,
    hyperparameter_grid=None,
    use_gridsearch=False,
    top_k_results=5,
    verbose=False
):
    results = []

    # Generate all possible non-empty combinations of features
    all_combinations = [comb for i in range(1, len(combinations_dict) + 1) 
                        for comb in itertools.combinations(combinations_dict.keys(), i)]

    print(f"\n🔍 Testing {len(all_combinations)} feature combinations for {model_name}...\n")

    for combination in all_combinations:
        # Select features
        selected_features = np.hstack([combinations_dict[feature] for feature in combination])

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            selected_features, y, test_size=0.2, random_state=42, stratify=y
        )

        # Preprocess features (Normalize, PCA if applicable)
        X_train, X_test = preprocess_features(
            X_train, X_test, 
            normalize=normalize, 
            apply_pca=apply_pca, 
            n_pca_components=n_pca_components,
            verbose=False  # Silence preprocessing output during the loop
        )

        # Optionally perform GridSearch
        if use_gridsearch:
            if hyperparameter_grid:
                print(f"\nPerforming grid search for combination: {combination}...")
                grid_search, best_params, best_score = grid_search_hyperparameter_tuning(
                    model=model,
                    param_grid=hyperparameter_grid,
                    X_train=X_train,
                    y_train=y_train
                )
                model_params = best_params
                print(f"Grid search complete. Best parameters: {best_params}, Best AUC: {best_score:.4f}")
            else:
                raise ValueError("Grid search enabled, but no hyperparameter grid provided.")
        elif not model_params:
            raise ValueError("Provide `model_params` or enable grid search.")

        # Cross-validation using K-Fold
        train_accuracies, val_accuracies, auc_scores = [], [], []
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        for train_idx, val_idx in kfold.split(X_train, y_train):
            fold_X_train, fold_X_val = X_train[train_idx], X_train[val_idx]
            fold_y_train, fold_y_val = y_train[train_idx], y_train[val_idx]

            # Initialize and train the model
            classifier = model.set_params(**model_params)
            classifier.fit(fold_X_train, fold_y_train)

            # Training and validation accuracy
            train_acc = accuracy_score(fold_y_train, classifier.predict(fold_X_train))
            val_acc = accuracy_score(fold_y_val, classifier.predict(fold_X_val))
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)

            # Calculate AUC if available
            if hasattr(classifier, "predict_proba"):
                y_val_proba = classifier.predict_proba(fold_X_val)
                auc_scores.append(roc_auc_score(fold_y_val, y_val_proba, multi_class='ovr'))

        # Calculate averages
        avg_train_accuracy = np.mean(train_accuracies)
        avg_val_accuracy = np.mean(val_accuracies)
        avg_auc = np.mean(auc_scores) if auc_scores else None
        overfitting_gap = avg_train_accuracy - avg_val_accuracy
        overfitting_status = "⚠️ Overfitting Risk" if (avg_train_accuracy == 1.0 or overfitting_gap > overfit_threshold) else "✅ No Overfitting"

        # Calculate combined score: 50% AUC + 50% validation accuracy
        combined_score = 0.5 * (avg_auc if avg_auc is not None else 0) + 0.5 * avg_val_accuracy

        # Store results
        results.append({
            'combination': combination,
            'average_auc': avg_auc,
            'average_train_accuracy': avg_train_accuracy,
            'average_val_accuracy': avg_val_accuracy,
            'combined_score': combined_score,
            'overfitting_status': overfitting_status,
            'overfitting_gap': overfitting_gap
        })

        if verbose:
            print(f"Combination: {combination}, "
                  f"AUC: {f'{avg_auc:.4f}' if avg_auc is not None else 'N/A'}, "
                  f"Train Acc: {avg_train_accuracy:.4f}, "
                  f"Val Acc: {avg_val_accuracy:.4f}, "
                  f"Combine Score: {combined_score:.4f}, "
                  f"Overfitting: {overfitting_status}")

    # Filter and sort by combined score
    sorted_results = sorted(results, key=lambda x: x['combined_score'], reverse=True)

    # Display top-k results
    print(f"\n📊 Top {top_k_results} Results (sorted by Combined Score):")
    for i, res in enumerate(sorted_results[:top_k_results], start=1):
        print(f"  {i}. Combination: {res['combination']}, "
              f"Combined Score: {res['combined_score']:.4f}, "
              f"Test Acc: {res['average_val_accuracy']:.4f}, "
              f"Train Acc: {res['average_train_accuracy']:.4f}, "
              f"AUC: {f'{res['average_auc']:.4f}' if res['average_auc'] is not None else 'N/A'}, "
              f"Train-Test Gap: {res['overfitting_gap']:.4f}, "
              f"Overfitting Status: {res['overfitting_status']}")

    return sorted_results


def evaluate_kmeans_feature_groups(feature_groups_selection, loaded_data, k=10, use_kmeans_pp=True, n_init=30, apply_pca_options=[True, False], verbose=True):
    results_summary = []

    for group_name, feature_selection in feature_groups_selection.items():
        for apply_pca in apply_pca_options:  
            if verbose:
                print(f"\n🔄 Testing Feature Group: {group_name} with PCA={'Yes' if apply_pca else 'No'}")
            
            selected_features, X_train, X_test, y_train, y_test, selected_features_names = load_and_split_features(
                loaded_data, feature_selection
            )

            X_full = np.vstack([X_train, X_test])
            y_full = np.hstack([y_train, y_test])

            # Preprocess features
            X_full, _ = preprocess_features(
                X_full, X_full, normalize=True, apply_pca=apply_pca, n_pca_components=0.95, verbose=False
            )

            # Choose KMeans++ initialization if specified
            kmeans = KMeans(
                n_clusters=k, 
                random_state=42, 
                n_init=n_init, 
                init='k-means++' if use_kmeans_pp else 'random'
            )
            cluster_labels = kmeans.fit_predict(X_full)

            # Calculate Silhouette and ARI scores
            silhouette = silhouette_score(X_full, cluster_labels)
            ari_score = adjusted_rand_score(y_full, cluster_labels)

            # Calculate majority class metrics
            percentage_majority_class = {}
            weighted_accuracy_sum = 0
            purity_numerator = 0
            total_samples = len(y_full)

            if verbose:
                print("\n📊 Cluster Statistics:")
            for cluster in np.unique(cluster_labels):
                assigned_class = mode(y_full[cluster_labels == cluster], keepdims=True).mode[0]
                total_samples_in_cluster = np.sum(cluster_labels == cluster)
                majority_class_count = np.sum((cluster_labels == cluster) & (y_full == assigned_class))
                percentage = (majority_class_count / total_samples_in_cluster) * 100
                percentage_majority_class[cluster] = percentage
                weighted_accuracy_sum += (majority_class_count / total_samples)
                purity_numerator += majority_class_count

                # Print cluster-level details
                if verbose:
                    print(f"Cluster {cluster} (Majority Class: {assigned_class}) → {percentage:.2f}% of samples belong to class {assigned_class}")

            # Calculate overall metrics
            mean_majority_percentage = np.mean(list(percentage_majority_class.values()))
            purity_score = purity_numerator / total_samples

            if verbose:
                print(f"\nMean Average of Majority Class Percentages: {mean_majority_percentage:.2f}%")
                print(f"Weighted Accuracy: {weighted_accuracy_sum:.4f}")
                print(f"Purity Score: {purity_score:.4f}")

            # Store results
            results_summary.append({
                'Feature Group': group_name,
                'PCA': 'Yes' if apply_pca else 'No',
                'KMeans Ini.': 'KMeans++' if use_kmeans_pp else 'Random',
                'Sil. Score': silhouette,
                'ARI Score': ari_score,
                'MMP': mean_majority_percentage,
                'Weighted Acc.': weighted_accuracy_sum,
                'Purity Score': purity_score
            })

    # Convert results to a DataFrame and display
    results_df = pd.DataFrame(results_summary)

    # Update the filename based on the KMeans initialization
    filename = "feature_group_results_kmeanspp.csv" if use_kmeans_pp else "feature_group_results_kmeans.csv"

    print("\n📊 Summary of Results:")
    print(results_df)

    # Save results to a meaningful CSV filename
    results_df.to_csv(filename, index=False)
    print(f"\n✅ Results saved to {filename}")

    return results_df