# # Import the necessary modules
# import os
# import csv
# import numpy as np

# # Define the path to the folders
# path = "A:/MSc/Dissertation (CS958)/multistream-audiovisual-speech-recognition/29May2023/Sheet"
# label_path = "A:/MSc/Dissertation (CS958)/lrs2_v1/mvlrs_v1/main"

# # Define the length of lip features
# lip_feature_length = 7

# # Initialize empty lists to store the data and labels
# data = []
# labels = []
# info_dict = {}

# # Initialize an empty list to store the tuples of directory name and word features
# dir_word_features = []

# # Loop through all the folders in the path
# for folder in os.listdir(path):
#     # Get the full path of the folder
#     folder_path = os.path.join(path, folder)

#     # Create an empty array to store the word features with shape (number of frames, lip_feature_length)
#     word_features = np.empty((0, lip_feature_length))

#     # Loop through all the csv files in the folder
#     for file in os.listdir(folder_path):
#         # Get the full path of the file
#         file_path = os.path.join(folder_path, file)
#         # Open the file as a csv reader
#         with open(file_path, 'r') as f:
#             reader = csv.reader(f)
#             # Loop through each row in the file
#             for row in reader:
#                 # print(row)
#                 # Check if the row has two elements
#                 if len(row) == 2:
#                     # Get the key and value from the row
#                     key = row[0]
#                     value = float(row[1])
#                     # Store the key and value in the info dictionary
#                     info_dict[key] = value
#                 else:
#                     # If the row does not have two elements, skip it or handle it differently
#                     pass

#             # Get the frame features from the info dictionary using the keys that match your requirement
#             width = info_dict['Box_width']
#             height = info_dict['Box_height']
#             area = info_dict['Final_area']
#             x_value = info_dict['Centroid_x']
#             y_value = info_dict['Centroid_y']
#             intensity = info_dict['intensity']
#             orientation = info_dict['orientation']
        
#             # Create a list of frame features
#             frame_features = [width, height, area, x_value, y_value, intensity, orientation]

#             # Append a tuple of folder name and word features to the dir_word_features list as a nested list instead of a numpy array to match your requirement
#             dir_word_features.append((folder, frame_features))

#             # Append the frame features to the word features array
#             word_features = np.append(word_features, [frame_features], axis=0)

#     # Append a tuple of folder name and word features to the data list 
#     # data.append((folder, word_features))
#     data.append(word_features)

#     # Get the label name from the folder name by splitting it on underscore and store them under folder_name and video_name lists
#     folder_name = folder.split('_')
#     # Get label from the dataset by using the video name and folder name with the label_path
#     text_file = label_path + '/' + folder_name[0] + '/' + folder_name[1] + '.txt'
#     with open(text_file, 'r') as t:
#         transcript = t.read()
#         # Get the label name from the transcript by splitting it on newline and store them under label_name list
#         label_name = transcript.split('\n')[0]
#     # Append the label name to the labels list
#     labels.append(label_name)
#     # Remove duplicates from the labels list
#     labels = list(dict.fromkeys(labels))

# # # Find the maximum number of frames among all words in data list 
# # max_frame_length = np.max([arr.shape[0] for _, arr in data])

# # # Loop through each tuple in data list 
# # for i in range(len(data)):
# #     # Get the directory name and word features from each tuple 
# #     dir_name, word_features = data[i]
# #     # Pad each word with zero value arrays to have the same number of frames as max_frame_length
# #     # word_features = np.pad(word_features, ((0,max_frame_length - word_features.shape[0]), (0,0)), mode='constant')
# #     # Replace each tuple in data list with the padded word features array 
# #     data[i] = word_features



# # Convert the data and labels lists into numpy arrays
# data = np.array(data)
# labels = np.array(labels)

# # Save the data and labels arrays into .npy files
# np.save('A:/MSc/Dissertation (CS958)/multistream-audiovisual-speech-recognition/data_without_padding.npy', data)
# np.save('A:/MSc/Dissertation (CS958)/multistream-audiovisual-speech-recognition/label.npy', labels)

# # Print a message to indicate success
# print('Data and labels saved successfully!')





import os
import pandas as pd
import numpy as np

# Specify the directory where your CSV files are
sheet_directory = 'A:/MSc/Dissertation (CS958)/multistream-audiovisual-speech-recognition/Sheet-Archie'
output_root_directory = 'A:/MSc/Dissertation (CS958)/lrs2_v1/mvlrs_v1/main'

# File to store the list of processed directories
processed_dirs_file = 'processed_dirs.txt'

# Load the list of processed directories
if os.path.exists(processed_dirs_file):
    with open(processed_dirs_file, 'r') as f:
        processed_dirs = f.read().splitlines()
else:
    processed_dirs = []

# Walk through all subdirectories in the sheet directory
for subdir, dirs, files in os.walk(sheet_directory):
    # Skip the directory if it has been processed
    if subdir in processed_dirs:
        continue

    csv_files = [f for f in files if f.endswith('.csv')]
    if csv_files:
        # Load each csv file as a numpy array and add to a list
        data_arrays = []
        for f in csv_files:
            df = pd.read_csv(os.path.join(subdir, f), header=None)
            if not df.empty:
                # Extract the first 7 values from the second column only and convert it to a numpy array
                data = df.iloc[:7, 1].to_numpy()

                # Check the size of each array
                if len(data) == 7:  # We expect exactly 7 entries
                    data_arrays.append(data)
                else:
                    print(f'Warning: Unexpected number of entries in csv file {f} in directory {subdir}.')
            else:
                print(f'Warning: DataFrame created from csv file {f} in directory {subdir} is empty.')

        # Concatenate all numpy arrays into a single 2D numpy array
        if data_arrays:
            data = np.vstack(data_arrays)
            print(f'Data shape: {data.shape}, Data type: {data.dtype}')  # Diagnostic output

            # Construct the output directory path
            subdir_name = os.path.basename(subdir)
            folder_name, video_name = subdir_name.split('_')
            output_dir = os.path.join(output_root_directory, folder_name)

            # Create the output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Construct the output file path
            output_path = os.path.join(output_dir, f'{video_name}_gabor.npy')

            # Save the 2D numpy array to a .npy file
            np.save(output_path, data)

            # Add the directory to the list of processed directories
            processed_dirs.append(subdir)

            # Update the processed directories file
            with open(processed_dirs_file, 'w') as f:
                for dir in processed_dirs:
                    f.write(dir + '\n')
