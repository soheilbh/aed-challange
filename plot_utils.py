import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
from scipy import signal

# Function to play an audio file in a Jupyter notebook
def play_sound(sound_data, sample_rate):
    return ipd.display(ipd.Audio(data=sound_data, rate=sample_rate))

# Function to create a histogram subplot for amplitude distribution
def create_histogram_subplot(ax, sound_data, filename, bins_list):
    histograms = {}
    for i, bins in enumerate(bins_list):
        hist, _ = np.histogram(sound_data, bins=bins, density=True)  # Compute histogram
        histograms[f"bins_set_{i+1}"] = hist  # Store histogram data
        
        # Plot histogram with step-pre style
        ax.plot(bins[:-1], hist, label=f'Bin Set {i+1}', alpha=0.7, drawstyle='steps-pre', picker=True, animated=True)
    
    ax.set_title(f'Histogram (Multiple Bin Sets)\n{filename}')
    ax.set_xlabel("Amplitude")
    ax.set_ylabel("Frequency")
    ax.legend()
    return histograms  # Return histogram data for feature extraction

# Define bin ranges for histograms
bins_set1 = np.linspace(-32768, 32767, 51)  # Fine granularity (50 bins)
bins_set2 = np.linspace(-32768, 32767, 21)  # Moderate granularity (20 bins)
bins_set3 = np.linspace(-32768, 32767, 11)  # Coarse granularity (10 bins)
bins_list = [bins_set1, bins_set2, bins_set3]  # Store all bin sets

# Function to plot the waveform of a sound file
def create_waveform_subplot(ax, sample_rate, sound_data, filename):
    time = np.linspace(0., len(sound_data) / sample_rate, len(sound_data))
    ax.plot(time, sound_data)
    ax.set_title(f'Waveform\n{filename}')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")

# Function to plot the spectrogram of a sound file
def create_spectrogram_subplot(ax, sample_rate, sound_data, filename):
    window = signal.windows.hann(1024)
    f, t, Sxx = signal.spectrogram(sound_data, sample_rate, window=window)
    im = ax.pcolormesh(t, f, 20 * np.log10(Sxx), shading='gouraud')  # Convert to dB scale
    ax.set_title(f'Spectrogram\n{filename}')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    return im  # Return spectrogram image for optional colorbar

# Function to visualize and analyze sounds by category
def visualize_category_sounds(wave_list_data):
    
    categories = []
    for cat, _, _, _ in wave_list_data:
        if cat not in categories:
            categories.append(cat)
    
    all_histograms = {}  # Store histograms for all categories
    
    for cat in categories:
        category_waves = [wave for wave in wave_list_data if wave[0] == cat]
        num_files = len(category_waves)
        
        plt.figure(figsize=(5 * num_files, 12))
        plt.suptitle(f'{cat} Category Analysis', fontsize=16)
        
        print(f"Analyzing {cat} category...")

        category_histograms = []
        for idx, (cat, filename, sample_rate, sound_data) in enumerate(category_waves):
            play_sound(sound_data, sample_rate)  # Play audio in notebook
            
            # Waveform plot
            ax_wave = plt.subplot(3, num_files, idx + 1)
            create_waveform_subplot(ax_wave, sample_rate, sound_data, filename)
            
            # Spectrogram plot
            ax_spec = plt.subplot(3, num_files, num_files + idx + 1)
            im_spec = create_spectrogram_subplot(ax_spec, sample_rate, sound_data, filename)
            
            # Histogram plot
            ax_hist = plt.subplot(3, num_files, 2 * num_files + idx + 1)
            histograms = create_histogram_subplot(ax_hist, sound_data, filename, bins_list)
            category_histograms.append(histograms)
        
        all_histograms[cat] = category_histograms  # Store histograms for this category
        plt.tight_layout()
        plt.show()
        
    return all_histograms  # Return all histograms for further analysis

def plot_coverage_radar(knn_results, svm_results, rf_results):
    # Extract performance metrics
    performance_data = {
        'KNN': [
            knn_results['avg_train_accuracy'],
            knn_results['avg_test_accuracy'],
            knn_results['avg_auc'],
            knn_results['avg_tpr'],
            knn_results['avg_fpr'],
            knn_results['avg_precision'],
            knn_results['avg_recall'],
            knn_results['avg_f1_score']
        ],
        'SVM': [
            svm_results['avg_train_accuracy'],
            svm_results['avg_test_accuracy'],
            svm_results['avg_auc'],
            svm_results['avg_tpr'],
            svm_results['avg_fpr'],
            svm_results['avg_precision'],
            svm_results['avg_recall'],
            svm_results['avg_f1_score']
        ],
        'Random Forest': [
            rf_results['avg_train_accuracy'],
            rf_results['avg_test_accuracy'],
            rf_results['avg_auc'],
            rf_results['avg_tpr'],
            rf_results['avg_fpr'],
            rf_results['avg_precision'],
            rf_results['avg_recall'],
            rf_results['avg_f1_score']
        ]
    }

    # Labels for the performance metrics
    metrics = ['Train Accuracy', 'Test Accuracy', 'AUC', 'TPR', 'FPR', 'Precision', 'Recall', 'F1-Score']

    # Radar plot setup
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    # Create the radar plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Plot each model's performance
    for model, values in performance_data.items():
        values += values[:1]  # Close the loop for each model
        ax.plot(angles, values, label=model)
        ax.fill(angles, values, alpha=0.25)

    # Format the plot
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10, fontweight='bold')
    ax.set_yticklabels([])  # Hide radial labels for simplicity
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title("Coverage Plot - Comparison of Model Performance", fontsize=14, fontweight='bold')
    plt.show()


def plot_model_performance_comparison(knn_results, svm_results, rf_results):
    # Extract metrics for models
    models = ['KNN', 'SVM', 'RF']
    test_accuracies = [
        knn_results['avg_test_accuracy'], 
        svm_results['avg_test_accuracy'], 
        rf_results['avg_test_accuracy']
    ]
    avg_tprs = [
        knn_results['avg_tpr'], 
        svm_results['avg_tpr'], 
        rf_results['avg_tpr']
    ]
    avg_fprs = [
        knn_results['avg_fpr'], 
        svm_results['avg_fpr'], 
        rf_results['avg_fpr']
    ]
    avg_aucs = [
        knn_results['avg_auc'], 
        svm_results['avg_auc'], 
        rf_results['avg_auc']
    ]
    precisions = [
        knn_results['avg_precision'], 
        svm_results['avg_precision'], 
        rf_results['avg_precision']
    ]
    recalls = [
        knn_results['avg_recall'], 
        svm_results['avg_recall'], 
        rf_results['avg_recall']
    ]
    f1_scores = [
        knn_results['avg_f1_score'], 
        svm_results['avg_f1_score'], 
        rf_results['avg_f1_score']
    ]

    # X-axis positions for models
    x = np.arange(len(models))

    # Create a figure and axis
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Bar plot for test accuracies
    bars = ax1.bar(x, test_accuracies, color='skyblue', alpha=0.7, label='Test Accuracy', width=0.4)
    
    # Set x-axis labels and title
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.set_title('Model Performance Comparison', fontsize=16)
    ax1.set_ylabel('Accuracy & Scores')
    ax1.set_ylim(0, 1.2)  # To give space for lines
    
    # Line plots for TPR, FPR, and AUC
    ax2 = ax1.twinx()
    ax2.plot(x, avg_tprs, marker='o', linestyle='--', color='green', label='TPR (Coverage)', linewidth=2)
    ax2.plot(x, avg_fprs, marker='o', linestyle='--', color='red', label='FPR (False Positives)', linewidth=2)
    ax2.plot(x, avg_aucs, marker='o', linestyle='-', color='orange', label='AUC', linewidth=2)
    ax2.plot(x, precisions, marker='o', linestyle='-', color='purple', label='Precision', linewidth=2)
    ax2.plot(x, recalls, marker='o', linestyle='-', color='brown', label='Recall', linewidth=2)
    ax2.plot(x, f1_scores, marker='o', linestyle='-', color='darkblue', label='F1-Score', linewidth=2)
    ax2.set_ylabel('Scores')

    # Adding value labels on bars
    for bar, acc in zip(bars, test_accuracies):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03, f'{acc:.2f}', ha='center', va='bottom')

    # Legends and layout adjustment
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.85))
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Show the plot
    plt.show()