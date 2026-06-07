import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("helmet_model_2class.keras")

img = cv2.imread("image.png")

img = cv2.resize(img, (224,224))
img = img / 255.0
img = np.expand_dims(img, axis=0)

prediction = model.predict(img, verbose=0)

confidence = prediction[0][0]

print(prediction)

if confidence > 0.5:
    print("Person No Helmet")
    print("Confidence:", round(confidence * 100, 2), "%")
else:
    print("Helmet")
    print("Confidence:", round((1 - confidence) * 100, 2), "%")