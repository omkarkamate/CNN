import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("helmet_model_2class.keras")

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    img = cv2.resize(frame, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)

    confidence = prediction[0][0]

    if confidence > 0.5:
        label = f"NO HELMET ({confidence*100:.1f}%)"
        color = (0, 0, 255)
    else:
        label = f"HELMET ({(1-confidence)*100:.1f}%)"
        color = (0, 255, 0)

    cv2.putText(
        frame,
        label,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.imshow("Helmet Detection", frame)

    key = cv2.waitKey(1)

    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()