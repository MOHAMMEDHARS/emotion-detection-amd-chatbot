import tensorflow as tf
import numpy as np
import cv2
from flask import Flask, render_template, Response, redirect

app = Flask(__name__)
model = tf.keras.models.load_model('emotion_detection_model.h5')

# Define emotion labels
emotions = ["Angry", "Disgusted", "Fearful", "Happy", "Sad", "Surprised", "Neutral"]

# Flag to indicate if happy emotion is detected
happy_detected = False

# Function to detect emotion from image
def detect_emotion(image):
    # Preprocess the image
    image = cv2.resize(image, (48, 48))
    image = np.expand_dims(image, axis=0)
    # Predict emotion
    predictions = model.predict(image)
    emotion_label = emotions[np.argmax(predictions)]
    return emotion_label

# Function to capture video from camera
def gen_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Flip the frame horizontally for better presentation
            frame = cv2.flip(frame, 1)
            # Display emotion for the captured frame
            emotion = detect_emotion(frame)
            cv2.putText(frame, emotion, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture')
def capture():
    global happy_detected
    if happy_detected:
        return redirect('/')
    cap = cv2.VideoCapture(0)
    success, frame = cap.read()
    if not success:
        print("Error: Could not capture frame.")
        return "Error: Could not capture frame."
    # Flip the frame horizontally for better presentation
    frame = cv2.flip(frame, 1)
    # Display emotion for the captured frame
    emotion = detect_emotion(frame)
    cv2.putText(frame, emotion, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    cv2.imwrite('captured_frame.jpg', frame)
    # Set flag if emotion is happy
    if emotion.lower() == 'sad':
        happy_detected = True
    return redirect('/')

if __name__ == "__main__":
    app.run(debug=True)
