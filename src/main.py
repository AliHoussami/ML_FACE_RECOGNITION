import cv2
import pickle
import numpy as np
from face_detection import FaceDetector
import dlib

# Paths to models
SHAPE_PREDICTOR_PATH = "../models/shape_predictor_68_face_landmarks.dat"
FACE_RECOGNITION_MODEL_PATH = "../models/dlib_face_recognition_resnet_model_v1.dat"

# Load known faces
KNOWN_FACES_FILE = "../data/known_faces.pkl"
try:
    with open(KNOWN_FACES_FILE, "rb") as file:
        known_faces = pickle.load(file)
except FileNotFoundError:
    print(f"Error: Known faces file not found at {KNOWN_FACES_FILE}.")
    exit()

# Initialize FaceDetector
detector = FaceDetector(SHAPE_PREDICTOR_PATH, FACE_RECOGNITION_MODEL_PATH)

# Initialize Dlib's frontal face detector for secondary detection
face_detector = dlib.get_frontal_face_detector()

def recognize_face(face_descriptor, known_faces, threshold=0.6):
    """Compare a face descriptor with known faces."""
    best_match = None
    min_distance = float("inf")

    for name, known_descriptor in known_faces.items():
        distance = np.linalg.norm(np.array(face_descriptor) - np.array(known_descriptor))
        if distance < min_distance and distance < threshold:
            min_distance = distance
            best_match = name

    return best_match

# Open webcam
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not camera.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Starting webcam... Press 'q' to quit.")
while True:
    ret, frame = camera.read()
    if not ret:
        break

    # Resize frame for faster processing
    frame = cv2.resize(frame, (320, 240))
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using the FaceDetector
    faces = detector.detect_faces(frame)

    # Detect faces using Dlib's frontal face detector
    dlib_faces = face_detector(gray_frame)

    # Draw frames for detected faces
    for face_rect in dlib_faces:
        x, y, w, h = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Recognize faces
    for face in faces:
        descriptor = face["descriptor"]
        name = recognize_face(descriptor, known_faces)

        bbox = face["bounding_box"]
        if name:
            cv2.putText(frame, name, (bbox["left"], bbox["top"] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            print(f"Recognized: {name}")
        else:
            cv2.putText(frame, "Unknown", (bbox["left"], bbox["top"] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("Live Face Recognition", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
