import matplotlib.pyplot as plt
import numpy as np
import pip
from scipy.signal import hilbert
import os

folder_path = "C:\\Users\\DELL\\\Desktop\\EOG Convert\\Horizontal left"  # Replace with the folder path containing your signal files
save_folder = "D:\\Users\\user\\Desktop\\Graduation Pro\\Hilbert transform\\Horizontal left"  # Replace with desired folder path to save the images

# Create the save folder if it does not exist
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Loop through files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):  # Filter for txt files
        # Load signal from file
        file_path = os.path.join(folder_path, filename)
        signal = np.loadtxt(file_path)

        # Compute Hilbert transform
        hilbert_transform = hilbert(signal)

        # Extract envelope of the Hilbert transform
        envelope = np.abs(hilbert_transform)

        # Reshape the envelope into an image
        image_data = np.reshape(envelope, (-1, len(signal)))

        # Plot Hilbert transform as an image without scaling
        plt.figure()
        plt.imshow(image_data, cmap='jet', aspect='auto')
        plt.axis('off')
        plt.title('Hilbert transform')
        # Save the plot as an image
        save_filename = os.path.splitext(filename)[0] + "_hilbert_envelope.png"  # Add prefix to the original filename
        save_path = os.path.join(save_folder, save_filename)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

        # Close the plot
        plt.close()

print("Hilbert transform envelopes saved as images in folder:", save_folder)