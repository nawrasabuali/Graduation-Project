import matplotlib.pyplot as plt
import numpy as np
import pip
from scipy.signal import stft
import os


# Path to the folder containing CSV files
folder_path = "C:\\Users\\DELL\\Desktop\\new"  # Replace with the folder path containing your signal files
save_folder = "C:\\Users\\DELL\\Desktop\\outdataa\\s"  # Replace with desired folder path to save the images

# Create the save folder if it does not exist
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Loop through files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):  # Filter for txt files
        # Load signal from file
        file_path = os.path.join(folder_path, filename)
        signal = np.loadtxt(file_path)

        # Compute STFT
        f, t, Zxx = stft(signal, nperseg=128)

        # Compute the magnitude of the STFT
        mag = np.abs(Zxx)

        # Reshape the magnitude into an image
        image_data = np.reshape(mag, (-1, len(t)))

        # Plot STFT as an image without scaling
        plt.figure()
        plt.imshow(image_data, cmap='jet', aspect='auto')
        plt.axis('off')
        plt.title('STFT')
        
        # Save the plot as an image
        save_filename = os.path.splitext(filename)[0] + "_stft_magnitude.png"  # Add prefix to the original filename
        save_path = os.path.join(save_folder, save_filename)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

        # Close the plot
        plt.close()

print("STFT magnitudes saved as images in folder:", save_folder)