import os
import numpy as np
from PIL import Image
from keras.api.models import Sequential
from keras.api.layers import Input, Conv2D, Dense, Flatten, Dropout, MaxPooling2D
from keras.api.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Initialize data and labels
data = []  # input
labels = []  # output
classes = 52
print("Total classes detected:", classes)
cur_path = os.getcwd()  # to get the current directory

# Dictionary of classes
classs = {
    0: 'Speed limit (5km/h)', 
            1: 'Speed limit (15km/h)',
            2: 'Speed limit (30km/h)', 
            3: 'Speed limit (40km/h)', 
            4: 'Speed limit (50km/h)', 
            5: 'Speed limit (60km/h)', 
            6: 'Speed limit (70km/h)', 
            7: 'speed limit (80km/h)', 
            8: 'Dont Go straight or left', 
            9: 'Unknown7', 10: 'Dont Go straight', 
            11: 'Dont Go Left', 
            12: 'Dont Go Left or Right', 
            13: 'Dont Go Right', 
            14: 'Dont overtake from Left', 
            15: 'No Uturn', 
            16: 'No Car', 
            17: 'No horn', 
            18: 'No entry', 
            19: 'No stopping', 
            20: 'Go straight or right', 
            21: 'Go straight', 
            22: 'Go Left', 
            23: 'Go Left or right',
            24: 'Go Right', 
            25: 'keep Left', 
            26: 'keep Right', 
            27: 'Roundabout mandatory', 
            28: 'watch out for cars', 
            29: 'Horn', 
            30: 'Bicycles crossing', 
            31: 'Uturn', 
            32: 'Road Divider', 
            33: 'Unknown6', 
            34: 'Danger Ahead', 
            35: 'Zebra Crossing', 
            36: 'Bicycles crossing', 
            37: 'Children crossing', 
            38: 'Dangerous curve to the left', 
            39: 'Dangerous curve to the right', 
            40: 'Unknown1', 41: 'Unknown2', 
            42: 'Unknown3', 43: 'Go right or straight', 
            44: 'Go left or straight', 45: 'Unknown4', 
            46: 'ZigZag Curve', 47: 'Train Crossing', 
            48: 'Under Construction', 
            49: 'Unknown5', 50: 'Fences', 
            51: 'Heavy Vehicle Accidents'
}

print("Obtaining Images & their Labels ...........")
for i in range(classes):
    paths = os.path.join(cur_path, 'DataSets/TRAIN/', str(i))
    images = os.listdir(paths)

    for a in images:
        try:
            image = Image.open(os.path.join(paths, a))
            image = image.resize((30, 30))  # Resize to 30x30 pixels
            image = image.convert('RGB')  # Ensure the image has 3 channels (RGB)
            image = np.array(image)
            data.append(image)
            labels.append(i)
            print(f"{a} Loaded")
        except Exception as e:
            print(f"Error loading image {a}: {e}")
            continue  # Skip this image and continue with the next

print("Dataset loaded")

# Convert lists to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Convert labels to one hot encoding
y_train = to_categorical(y_train, classes)
y_test = to_categorical(y_test, classes)

# Define and train a new model
print("Training Under process...")

model = Sequential()
model.add(Input(shape=x_train.shape[1:]))
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=x_train.shape[1:]))
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(classes, activation='softmax'))
print("Initialized model")

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test))

# Save the new model
model.save("myModel.h5")

# Plotting accuracy and loss
plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title("Accuracy")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
plt.savefig("Accuracy.png")

plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title("LOSS")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()
plt.savefig("LOSS.png")

print("Saved Model and graph to disk")
