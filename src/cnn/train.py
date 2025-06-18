from src.model import LeNet
from keras.preprocessing.image import ImageDataGenerator

model = LeNet()

for name, param in model.named_parameters():
    print(name, param.size(), param.requires_grad)

train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('',
                                                 target_size = (64, 64),
                                                 batch_size = 32)

test_set = test_datagen.flow_from_directory('',
                                            target_size = (64, 64),
                                            batch_size = 32)

classifier = model.fit_generator(training_set,
                         steps_per_epoch = 80,
                         epochs = 10,
                         verbose=1,
                         validation_data = test_set,
                         validation_steps = 2)

classifier.save("trained_weights.h5")
print("Saved model to disk")