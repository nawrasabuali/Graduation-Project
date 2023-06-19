import pandas as pd
import scipy.io

# Set the path to the input and output folders
input_folder = 'C:\\Users\\user\\Desktop\\EOG \\Vertical  Up'
output_folder = 'C:\\Users\\user\\Desktop\\EOG Convert\\Vertical  Up'

# Loop through each file in the input folder
for filename in os.listdir(input_folder):
    # Check if the file is a ".mat" file
    if filename.endswith('.mat'):
        # Load the data from the file
        mat_data = scipy.io.loadmat(os.path.join(input_folder, filename))
        # Convert the data to a Pandas DataFrame
        df = pd.DataFrame(mat_data['x'])
 # Set the output file path and filename
        output_filename = os.path.splitext(filename)[0] + '.csv'
        output_path = os.path.join(output_folder, output_filename)
# Write the DataFrame to a CSV file
        df.to_csv(output_path, index=False)