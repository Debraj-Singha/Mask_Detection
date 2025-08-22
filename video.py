from tensorflow.keras.models import load_model
import numpy as np
import cv2

# Load your trained model
# Ensure you are loading the correct model that you saved from your notebook.
# The notebook saves it as "mask_detector.keras".
model = load_model('mask_detector.keras') 

# Load OpenCV's Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face_mask(face):
    """
    Detects if a face is wearing a mask or not.

    Args:
        face: A cropped image of a face.

    Returns:
        A tuple containing the label ("with_mask" or "without_mask") and the
        probability of the prediction.
    """
    # Preprocess the face image for the model
    img = cv2.resize(face, (128, 128))
    img_array = np.array(img, dtype='float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension

    # Get the single probability output from the model
    prob = model.predict(img_array, verbose=0)[0][0]

    # The model predicts the probability of 'without_mask' (class 1)
    # If prob > 0.5, it's likely 'without_mask'
    # If prob <= 0.5, it's likely 'with_mask'
    if prob > 0.5:
        return "Without Mask", prob
    else:
        return "With Mask", 1 - prob # Return confidence for 'with_mask'


# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    for (x, y, w, h) in faces:
        # Crop the face from the frame
        face = frame[y:y+h, x:x+w]
        
        # Get the prediction from our model
        label, prob = detect_face_mask(face)

        # Determine the color for the bounding box and text
        color = (0, 255, 0) if label == "With Mask" else (0, 0, 255)
        
        # Draw the bounding box around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Display the label and confidence score
        text = f"{label}: {prob:.2f}"
        cv2.putText(frame, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Show the final frame with detections
    cv2.imshow("Mask Detection", frame)

    # Exit on 'x' key press
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()
