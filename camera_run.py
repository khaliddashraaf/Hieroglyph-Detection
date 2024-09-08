import cv2
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model('/content/drive/MyDrive/Egypt.h5')

import cv2
import numpy as np
from keras.models import load_model

# Initialize the camera capture object
cap = cv2.VideoCapture(0)

# Define the classes of Egyptian hieroglyphs
classes = ['class1', 'class2', 'class3', ..., 'class40']

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    # Check if the frame is valid
    if not ret:
        print("Error: Failed to capture frame from camera")
        break

    # Preprocess the frame to match the input size of the model
    frame = cv2.resize(frame, (224, 224))
    frame = np.expand_dims(frame, axis=0)

    # Make a prediction on the frame using the trained model
    prediction = model.predict(frame)
    prediction_class = np.argmax(prediction)

    # Display the predicted class on the frame
    class_text = classes[prediction_class]
    cv2.putText(frame, class_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('frame', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
