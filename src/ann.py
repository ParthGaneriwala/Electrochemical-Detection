import os

import cv2
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, LearningRateScheduler
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from tensorflow import keras
from tqdm import tqdm
from tensorflow.keras.optimizers import Adam, SGD

# Function to load data from text files
def load_dataset(dataset_path):
    data = []
    labels = []

    # Get the list of concentration folders
    concentration_folders = [folder for folder in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, folder))]

    for concentration_folder in tqdm(concentration_folders, desc="Loading Concentrations"):
        concentration_path = os.path.join(dataset_path, concentration_folder)
        print(concentration_folder)
        # Get the list of quality folders (good, not good)
        quality_folders = [folder for folder in os.listdir(concentration_path) if os.path.isdir(os.path.join(concentration_path, folder))]
        #
        for quality_folder in quality_folders:
            if(quality_folder == 'Good'):
                quality_path = os.path.join(concentration_path, quality_folder)
                print(quality_folder)

            # Check if the PNG folder exists
                txt_folder = os.path.join(quality_path, "txt")
                if os.path.exists(txt_folder):
                    # Get the list of PNG files
                    txt_files = [file for file in os.listdir(txt_folder) if file.endswith('.txt')]
                    print(txt_files)
                    for txt_file in txt_files:
                        txt_path = os.path.join(txt_folder, txt_file)

                        with open(txt_path, 'r') as f:
                            lines = f.readlines()
                            x_coords = []
                            y_coords = []

                            for line in lines:
                                # Split each line by comma
                                parts = line.strip().split(',')

                                # Convert parts to floating-point numbers
                                x = float(parts[0])
                                y = float(parts[1])

                                # Append x and y coordinates to lists
                                # x_coords.append(x)
                                y_coords.append(y)

                            # Append x and y coordinates as a tuple to data
                            data.append(y_coords)

                            # Append the label (concentration_folder) to labels
                            labels.append(float(concentration_folder.split('mM')[0]))
    return np.array(data), np.array(labels)

# Load dataset
dataset_path = 'D:\ElectroChemicalData\Trim_CdTxtfiles' # Change to your dataset folder path
data, labels = load_dataset(dataset_path)

# Determine the maximum length of x and y coordinate lists
max_length = max(len(y_coords) for y_coords in data)
min_length = min(len(y_coords) for y_coords in data)
print(min_length,max_length)
# Pad or truncate x and y coordinate lists to max_length
# padded_data = []
# for y_coords in data:
#     padded_y_coords = np.pad(y_coords, (0, max_length - len(y_coords)), mode='empty')
#     padded_data.append(padded_y_coords)
#
# # Convert padded_data to NumPy array
# reshaped_data = np.array(padded_data)
# print(reshaped_data, labels)



# Split the reshaped data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.3, random_state=42)
# Define the ANN model architecture
model = Sequential([
    Dense(128, activation='relu', input_shape=(240,)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
early_stopping = EarlyStopping(monitor='mae', patience=5, restore_best_weights=True)

# Define learning rate scheduler
def lr_scheduler(epoch, learning_rate):
    if epoch % 5 == 0 and epoch > 0:
        learning_rate = learning_rate * 0.5
    return learning_rate

lr_scheduler_callback = LearningRateScheduler(lr_scheduler)
optimizer = Adam(learning_rate=0.01)

model.compile(loss='mse', optimizer=optimizer, metrics=['mae','mse'])  # Use appropriate loss and optimizer for regression

# Train the model
history = model.fit(train_data, train_labels, epochs=100, batch_size=32, validation_split=0.2,  callbacks=[early_stopping, lr_scheduler_callback],verbose=1)
plt.style.use('seaborn')
plt.figure(figsize=(6,6))
plt.plot(history.history['loss'], color='b', label="training loss")
plt.plot(history.history['val_loss'], color='r', label="validation loss")
plt.legend()
plt.show()

plt.figure()

plt.figure(figsize=(6,6))
plt.plot(history.history['mae'], color='b', label="Mean absolute error")
plt.plot(history.history['val_mae'], color='r',label="Validation Mean absolute error")
plt.legend()
plt.show()
plt.figure(figsize=(6,6))
plt.plot(history.history['mse'], color='b', label="Mean squared error")
plt.plot(history.history['val_mse'], color='r',label="Validation Mean squared error")
plt.legend()
plt.show()
# Evaluate the model on the test set
test_loss, test_mae, test_mse= model.evaluate(test_data, test_labels)
print(f'Test Loss: {test_loss}')
print(f'Test MAE: {test_mae}')
print(f'Test MSE: {test_mse}')

# Make predictions
predictions = model.predict(test_data)
predictions_df = pd.DataFrame(np.ravel(predictions),columns=["Predictions"])
comparison_df = pd.concat([pd.DataFrame(test_labels,columns=["Real Values"]), predictions_df],axis=1)
print(comparison_df)
# # Visualize predictions against actual concentrations
# # Add visualization code here
