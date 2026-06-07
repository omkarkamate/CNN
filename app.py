import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dense

labels = {
    "Helmet": 0,
    "person_no_helmet": 1
}

X = []
y = []

for folder_name in labels:

    folder_path = os.path.join("Helmet_Dataset", folder_name)

    for file_name in os.listdir(folder_path):

        image_path = os.path.join(folder_path, file_name)

        img = cv2.imread(image_path)

        if img is None:
            continue

        img = cv2.resize(img, (224, 224))

        X.append(img)

        y.append(labels[folder_name])

X = np.array(X, dtype="float32")
y = np.array(y)

X = X / 255.0

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

model = Sequential()

model.add(Conv2D(32,(3,3),activation='relu',input_shape=(224,224,3)))

model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64,(3,3),activation='relu'))

model.add(MaxPooling2D((2,2)))

model.add(Conv2D(128,(3,3),activation='relu'))

model.add(MaxPooling2D((2,2)))

model.add(GlobalAveragePooling2D())

model.add(Dense(64,activation='relu'))

model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

history = model.fit(X_train,y_train,epochs=40,validation_data=(X_test, y_test))


model.save("helmet_model_2class.keras")