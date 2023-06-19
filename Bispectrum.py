import os
import numpy as np
from scipy.signal import detrend, stft
from scipy.special import comb
import matplotlib.pyplot as plt
from PIL import Image

def bispectrum(signal):
    f, t, Zxx = stft(signal, nperseg=128)
    Zxx = Zxx[:, :int(Zxx.shape[1]/2)+1]
    B = np.zeros((f.shape[0], f.shape[0]), dtype=np.float64)
    for i in range(f.shape[0]):
        for j in range(i+1):
            k = i-j
            B[i,j] = np.sum(Zxx[i,:]*Zxx[j,:]*np.conj(Zxx[k,:]))/Zxx.shape[1]
            B[j,i] = B[i,j]
    return B

def csv_to_bispectrum(csv_file):
    signal = np.loadtxt(csv_file, delimiter=',')
    signal = detrend(signal)
    B = bispectrum(signal)
    B = np.abs(B)
    B /= np.max(B)
    B = 255 * B
    B = B.astype(np.uint8)
    return B

folder_path = "C:\\Users\\DELL\\Desktop\\new"  # Replace with the folder path containing your signal files
save_folder = "C:\\Users\\DELL\\Desktop\\outdataa\\s"  # Replace with desired folder path to save the images

# Create the output folder if it doesn't exist yet
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

for csv_file in os.listdir(folder_path):
    if csv_file.endswith('.csv'):
        csv_path = os.path.join(folder_path, csv_file)
        image_path = os.path.join(save_folder, csv_file[:-4] + '.png')
        B = csv_to_bispectrum(csv_path)
        
        plt.imshow(B,cmap='jet', aspect='auto')
        plt.axis('off')
        plt.title('Bispectrum')
        plt.savefig(image_path, transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close()
       

        # Resize the image
        img = Image.open(image_path)
        new_width, new_height = 496, 369  # change to desired size
        img = img.resize((new_width, new_height))
        img.save(image_path)

print("Bispectrum saved as images in folder:", save_folder)
