import os
import numpy as np
import cv2
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
from matplotlib import pyplot as plt

# Function to load images and labels from the dataset folder
def load_dataset(dataset_path):
    images = []
    labels = []

    # Get the list of concentration folders
    concentration_folders = [folder for folder in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, folder))]

    for concentration_folder in tqdm(concentration_folders, desc="Loading Concentrations"):
        concentration_path = os.path.join(dataset_path, concentration_folder)

        # Get the list of quality folders (good, not good)
        quality_folders = [folder for folder in os.listdir(concentration_path) if os.path.isdir(os.path.join(concentration_path, folder))]

        for quality_folder in quality_folders:
            quality_path = os.path.join(concentration_path, quality_folder)

            # Check if the PNG folder exists
            png_folder = os.path.join(quality_path, "PNG")
            if os.path.exists(png_folder):
                # Get the list of PNG files
                png_files = [file for file in os.listdir(png_folder) if file.endswith('.png')]

                for png_file in png_files:
                    png_path = os.path.join(png_folder, png_file)

                    # Read the image
                    image = cv2.imread(png_path)

                    # Resize the image to fit the CNN model architecture
                    image = cv2.resize(image, (128, 128))

                    # Append the image and label to the lists
                    images.append(image)
                    labels.append(concentration_folder)

    return np.array(images), np.array(labels)

# Load dataset
dataset_path = 'D:\AI Project with Parth\Cu2+' # Change to your dataset folder path
images, labels = load_dataset(dataset_path)
print(labels)

# Preprocess images and labels
images = images.astype('float32') / 255.0
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
labels = to_categorical(labels)

# Split dataset into training and testing sets

train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2

# Define LeNet model architecture
model = Sequential()
model.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=(128,128,3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(120, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

# Define learning rate scheduler
def lr_scheduler(epoch, learning_rate):
    if epoch % 10 == 0 and epoch > 0:
        learning_rate = learning_rate * 0.9
    return learning_rate

lr_scheduler_callback = LearningRateScheduler(lr_scheduler)

# Define model checkpoint callback
checkpoint_callback = ModelCheckpoint(filepath='model_checkpoint.h5', monitor='val_accuracy', save_best_only=True, verbose=1)

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train the model
print("Training LeNet model...")
history = model.fit(train_images, train_labels, batch_size=32, epochs=100, validation_data=(val_images, val_labels), callbacks=[early_stopping, lr_scheduler_callback, checkpoint_callback], verbose=1)
#Evaluation
plt.style.use('seaborn')
plt.figure(figsize=(6,6))
plt.plot(history.history['loss'], color='b', label="training loss")
plt.plot(history.history['val_loss'], color='r', label="validation loss")
plt.legend()
plt.show()

plt.figure()

plt.figure(figsize=(6,6))
plt.plot(history.history['accuracy'], color='b', label="training accuracy")
plt.plot(history.history['val_accuracy'], color='r',label="validation accuracy")
plt.legend()
plt.show()

# Load the best model checkpoint
model.load_weights('model_checkpoint.h5')

# Function to visualize predictions on a subset of images
def visualize_predictions(images, true_labels, predicted_labels, num_samples=10):
    fig, axes = plt.subplots(nrows=num_samples, ncols=2, figsize=(10, 20))
    for i in range(num_samples):
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title(f"True: {label_encoder.inverse_transform([np.argmax(true_labels[i])])[0]}")
        axes[i, 0].axis('off')
        axes[i, 1].imshow(images[i])
        axes[i, 1].set_title(f"Predicted: {label_encoder.inverse_transform([np.argmax(predicted_labels[i])])[0]}")
        axes[i, 1].axis('off')
    plt.tight_layout()
    plt.show()

# Evaluate the model on the test set
print("Evaluating the model on the test set...")
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

# Make predictions on the test set
test_predictions = model.predict(test_images)

# Visualize predictions on a subset of test images
visualize_predictions(test_images, test_labels, test_predictions)
