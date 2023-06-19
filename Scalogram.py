import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.signal import chirp, find_peaks, peak_widths
from scipy.signal import cwt, morlet

folder_path = "C:\\Users\\DELL\\Desktop\\EOG Convert to .csv\\Vertical  Up" # Replace with the folder path containing your signal files
save_folder = "C:\\Users\\DELL\\Desktop\\Scalogram\\Vertical  Up_4"  # Replace with desired folder path to save the images

# Create the save folder if it does not exist
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Loop through files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):  # Filter for CSV files
        # Load signal from file
        file_path = os.path.join(folder_path, filename)
        signal = np.loadtxt(file_path)

        # Define the wavelet parameters
        widths = np.arange(1, len(signal)//2)  # Widths of the wavelets
        wavelet = morlet  # Wavelet function

        # Compute the scalogram
        scalogram = np.abs(cwt(signal, wavelet, widths))

        # Plot the scalogram
        plt.figure()
        plt.imshow(scalogram, cmap='jet', aspect='auto')
      
        plt.axis('off')
        # Save the plot as an image
        save_filename = os.path.splitext(filename)[0] + "_Scalogram.png"  # Add prefix to the original filename
        save_path = os.path.join(save_folder, save_filename)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        # Close the plot
        plt.close()

print("Scalograms saved as images in folder:", save_folder) # Plot the scalogram
     