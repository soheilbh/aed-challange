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
