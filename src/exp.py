import os
import numpy as np
import cv2
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
    class_folders = [folder for folder in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, folder))]
    for directory in tqdm(class_folders, desc="Loading Concentration"):
        class_path = os.path.join(dataset_path, directory)
        class_label = directory
        for file in os.listdir(class_path):
            for fi in os.listdir(os.path.join(class_path,file)):
                if fi == "PNG":
                    file_path = os.path.join(class_path, file, fi)
                    for f in os.listdir(file_path):
                        if f.endswith('.png'):
                            img_path = os.path.join(file_path, f)
                            image = cv2.imread(img_path)
                            image = cv2.resize(image, (64, 64)) # Resize image to fit LeNet architecture
                            images.append(image)
                            labels.append(file)
    return np.array(images), np.array(labels)

# Load dataset
dataset_path = 'D:\AI Project with Parth\Cd2+' # Change to your dataset folder path
images, labels = load_dataset(dataset_path)

# Preprocess images and labels
images = images.astype('float32') / 255.0
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
labels = to_categorical(labels)

# Split dataset into training and testing sets
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Define LeNet model architecture
model = Sequential()
model.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=(64,64,3)))
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

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train the model
print("Training LeNet model...")
history = model.fit(train_images, train_labels, batch_size=32, epochs=100, validation_data=(test_images, test_labels), callbacks=[early_stopping, lr_scheduler_callback], verbose=1)
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
# Evaluate the model
print("Evaluating the model...")
loss, accuracy = model.evaluate(test_images, test_labels)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')
