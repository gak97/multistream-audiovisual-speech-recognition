# Import the necessary modules
import os
import csv
import numpy as np

# Define the path to the folders
path = "A:/MSc/Dissertation (CS958)/multistream-audiovisual-speech-recognition/29May2023/Sheet"

# Define the max frame length of a word
max_frame_length = 20

# Define the length of lip features
lip_feature_length = 7

# Define the length of characters in label name
label_name_length = 24

# Initialize empty lists to store the data and labels
data = []
labels = []

# Loop through all the folders in the path
for folder in os.listdir(path):
    # Get the full path of the folder
    folder_path = os.path.join(path, folder)

    # Loop through all the csv files in the folder
    for file in os.listdir(folder_path):
        # Get the full path of the file
        file_path = os.path.join(folder_path, file)
        # Open the file as a csv reader
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            # Skip the header row
            next(reader)
            # Create an empty array to store the word features with shape (max_frame_length, lip_feature_length)
            word_features = np.zeros((max_frame_length, lip_feature_length))
            # Initialize a variable to store the current frame index
            frame_index = 0
            # Loop through each row in the file
            for row in reader:
                # Get the frame features from the row
                frame_features = [float(x) for x in row[1:8]]
                # If the frame index is less than the max frame length
                if frame_index < max_frame_length:
                    # Fill the word features array with the frame features at the current index
                    word_features[frame_index] = frame_features
                    # Increment the frame index by 1
                    frame_index += 1
                else:
                    # If the frame index is equal or greater than the max frame length, skip this row or handle it differently
                    pass
            # Append the word features to the data list
            data.append(word_features)
            # Create an empty array to store the label name with shape (label_name_length,) and itemsize 1 (one byte per character)
            label_name = np.chararray((label_name_length,), itemsize=1)
            # Initialize a variable to store the current label index
            label_index = 0
            # Loop through each character in the folder name
            for char in folder:
                # Check if the character is a digit
                if char.isdigit():
                    # Fill the label name array with the character at the current index
                    label_name[label_index] = char
                    # Increment the label index by 1
                    label_index += 1
                else:
                    # If the character is not a digit, skip it or handle it differently
                    pass
            # Append the label name to the labels list
            labels.append(label_name.decode())

# Convert the data and labels lists into numpy arrays
data = np.array(data)
labels = np.array(labels)

# Save the data and labels arrays into .npy files
np.save('A:/MSc/Dissertation (CS958)/multistream-audiovisual-speech-recognition/data.npy', data)
np.save('A:/MSc/Dissertation (CS958)/multistream-audiovisual-speech-recognition/label.npy', labels)

# Print a message to indicate success
print('Data and labels saved successfully!')
