# Import the necessary modules
import os
import csv
import numpy as np

# Define the path to the folders
path = "A:/MSc/Dissertation (CS958)/multistream-audiovisual-speech-recognition/29May2023/Sheet"
label_path = "A:/MSc/Dissertation (CS958)/lrs2_v1/mvlrs_v1/main"

# Define the length of lip features
lip_feature_length = 7

# Initialize empty lists to store the data and labels
data = []
labels = []
info_dict = {}

# Initialize an empty list to store the tuples of directory name and word features
dir_word_features = []

# Loop through all the folders in the path
for folder in os.listdir(path):
    # Get the full path of the folder
    folder_path = os.path.join(path, folder)

    # Create an empty array to store the word features with shape (number of frames, lip_feature_length)
    word_features = np.empty((0, lip_feature_length))

    # Loop through all the csv files in the folder
    for file in os.listdir(folder_path):
        # Get the full path of the file
        file_path = os.path.join(folder_path, file)
        # Open the file as a csv reader
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            # Loop through each row in the file
            for row in reader:
                # print(row)
                # Check if the row has two elements
                if len(row) == 2:
                    # Get the key and value from the row
                    key = row[0]
                    value = float(row[1])
                    # Store the key and value in the info dictionary
                    info_dict[key] = value
                else:
                    # If the row does not have two elements, skip it or handle it differently
                    pass

            # Get the frame features from the info dictionary using the keys that match your requirement
            width = info_dict['Box_width']
            height = info_dict['Box_height']
            area = info_dict['Final_area']
            x_value = info_dict['Centroid_x']
            y_value = info_dict['Centroid_y']
            intensity = info_dict['intensity']
            orientation = info_dict['orientation']
        
            # Create a list of frame features
            frame_features = [width, height, area, x_value, y_value, intensity, orientation]

            # Append a tuple of folder name and word features to the dir_word_features list as a nested list instead of a numpy array to match your requirement
            dir_word_features.append((folder, frame_features))

            # Append the frame features to the word features array
            word_features = np.append(word_features, [frame_features], axis=0)

    # Append a tuple of folder name and word features to the data list 
    data.append((folder, word_features))

    # Get the label name from the folder name by splitting it on underscore and store them under folder_name and video_name lists
    folder_name = folder.split('_')
    # Get label from the dataset by using the video name and folder name with the label_path
    text_file = label_path + '/' + folder_name[0] + '/' + folder_name[1] + '.txt'
    with open(text_file, 'r') as t:
        transcript = t.read()
        # Get the label name from the transcript by splitting it on newline and store them under label_name list
        label_name = transcript.split('\n')[0]
    # Append the label name to the labels list
    labels.append(label_name)
    # Remove duplicates from the labels list
    labels = list(dict.fromkeys(labels))

# Find the maximum number of frames among all words in data list 
max_frame_length = np.max([arr.shape[0] for _, arr in data])

# Loop through each tuple in data list 
for i in range(len(data)):
    # Get the directory name and word features from each tuple 
    dir_name, word_features = data[i]
    # Pad each word with zero value arrays to have the same number of frames as max_frame_length
    word_features = np.pad(word_features, ((0,max_frame_length - word_features.shape[0]), (0,0)), mode='constant')
    # Replace each tuple in data list with the padded word features array 
    data[i] = word_features

# Convert the data and labels lists into numpy arrays
data = np.array(data)
labels = np.array(labels)

# Save the data and labels arrays into .npy files
np.save('A:/MSc/Dissertation (CS958)/multistream-audiovisual-speech-recognition/data.npy', data)
np.save('A:/MSc/Dissertation (CS958)/multistream-audiovisual-speech-recognition/label.npy', labels)

# Print a message to indicate success
print('Data and labels saved successfully!')
