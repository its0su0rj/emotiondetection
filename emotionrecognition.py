import cv2
import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np

# Load pre-trained face cascade and emotion detection model
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
emotion_model = load_model("emotion_model.hdf5")

# Define the emotions
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces, gray

def detect_emotion(face_roi):
    # Resize the image to match the expected input shape of the model
    face_roi = cv2.resize(face_roi, (64, 64))
    
    # Convert to grayscale if needed
    if len(face_roi.shape) == 3:
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

    # Expand dimensions to create a batch size of 1
    face_roi = np.expand_dims(face_roi, axis=-1)
    face_roi = np.expand_dims(face_roi, axis=0)
    
    # Normalize the pixel values to be between 0 and 1
    face_roi = face_roi / 255.0

    emotion_pred = emotion_model.predict(face_roi)
    emotion_label = emotion_labels[np.argmax(emotion_pred)]
    return emotion_label

def main():
    st.title("Emotion Recognition System by Abhistkt")
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose a tab", ["Home", "Your Facial Emotion"])

    if app_mode == "Home":
        st.write("Welcome to the Emotion Recognition System!")
        st.write("Please select a tab from the sidebar.")
    elif app_mode == "Your Facial Emotion":
        st.write("Facial Emotion Recognition in Real Time")
        st.write("Press 'q' to stop the video stream.")
        show_camera()

def show_camera():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        faces, gray = detect_face(frame)

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            emotion = detect_emotion(face_roi)

            # Draw a rectangle around the face and display the emotion label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2, cv2.LINE_AA)

        # Convert the frame to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the resulting frame
        st.image(frame_rgb, channels="RGB", use_column_width=True)

        # Check if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture object
    cap.release()

    # Stop Streamlit app
    st.stop()

if __name__ == "__main__":
    main()
