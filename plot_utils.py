import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
from scipy import signal

def play_sound(sound_data, sample_rate):
    """
    Plays the sound in a Jupyter notebook cell using IPython.display.
    """
    return ipd.display(ipd.Audio(data=sound_data, rate=sample_rate))

def create_histogram_subplot(ax, sound_data, filename, bins_list):
    histograms = {}
    for i, bins in enumerate(bins_list):
        hist, _ = np.histogram(sound_data, bins=bins, density=True)  # Calculate histogram
        histograms[f"bins_set_{i+1}"] = hist  # Store histogram
        
        # Plot histograms for each bin set (overlayed)
        ax.plot(bins[:-1], hist, label=f'Bin Set {i+1}', alpha=0.7,drawstyle='steps-pre',picker=True, animated=True)
    
    ax.set_title(f'Histogram (Multiple Bin Sets)\n{filename}')
    ax.set_xlabel("Amplitude")
    ax.set_ylabel("Frequency")
    ax.legend()
    return histograms  # Return all histograms for feature extraction

bins_set1 = np.linspace(-32768, 32767, 51)  # 50 bins for detailed distribution analysis (higher granularity)
bins_set2 = np.linspace(-32768, 32767, 21)  # 20 bins for moderate granularity (balanced overview)
bins_set3 = np.linspace(-32768, 32767, 11)  # 10 bins for a broader overview (coarser granularity)
bins_list = [bins_set1, bins_set2, bins_set3]  # Combine all bin sets


def create_waveform_subplot(ax, sample_rate, sound_data, filename):
    """
    Plots the waveform on the given Axes object (ax).
    """
    time = np.linspace(0., len(sound_data) / sample_rate, len(sound_data))
    ax.plot(time, sound_data)
    ax.set_title(f'Waveform\n{filename}')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")

def create_spectrogram_subplot(ax, sample_rate, sound_data, filename):
    """
    Plots a spectrogram using scipy.signal.spectrogram on Axes ax.
    Returns im so you can potentially add a colorbar if needed.
    """
    window = signal.windows.hann(1024)
    f, t, Sxx = signal.spectrogram(sound_data, sample_rate, window=window)
    im = ax.pcolormesh(t, f, 20 * np.log10(Sxx), shading='gouraud')
    ax.set_title(f'Spectrogram\n{filename}')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    return im

def create_histogram_subplot(ax, sound_data, filename):
    """
    Plots a histogram of the amplitude values.
    """
    ax.hist(sound_data, bins=50)
    ax.set_title(f'Histogram\n{filename}')
    ax.set_xlabel("Amplitude")
    ax.set_ylabel("Count")

def visualize_category_sounds(wave_list_data):
    """
    Loops through wave_list_data, grouping by categories,
    and creates a figure showing Waveform, Spectrogram, and Histogram for each sample.
    """
    # Extract a unique list of categories in order
    categories = []
    for cat, _, _, _ in wave_list_data:
        if cat not in categories:
            categories.append(cat)
    
    # Iterate through each category
    for cat in categories:
        # Filter waves for this category
        category_waves = [wave for wave in wave_list_data if wave[0] == cat]
        
        # Determine how many files in this category
        num_files = len(category_waves)
        
        # Create a figure
        plt.figure(figsize=(5 * num_files, 12))
        plt.suptitle(f'{cat} Category Analysis', fontsize=16)
        
        print(f"Analyzing {cat} category...")
        
        # For each file in the category
        for idx, (cat, filename, sample_rate, sound_data) in enumerate(category_waves):
            # Play the audio in the notebook
            play_sound(sound_data, sample_rate)
            
            # Waveform subplot
            ax_wave = plt.subplot(3, num_files, idx + 1)
            create_waveform_subplot(ax_wave, sample_rate, sound_data, filename)
            
            # Spectrogram subplot
            ax_spec = plt.subplot(3, num_files, num_files + idx + 1)
            im_spec = create_spectrogram_subplot(ax_spec, sample_rate, sound_data, filename)
            
            # Histogram subplot
            ax_hist = plt.subplot(3, num_files, 2 * num_files + idx + 1)
            create_histogram_subplot(ax_hist, sound_data, filename)
        
        plt.tight_layout()
        plt.show()
