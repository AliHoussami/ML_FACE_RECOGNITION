import dlib
import cv2
import os
import pickle

# Paths to models
SHAPE_PREDICTOR_PATH = "../models/shape_predictor_68_face_landmarks.dat"
FACE_RECOGNITION_MODEL_PATH = "../models/dlib_face_recognition_resnet_model_v1.dat"

# Initialize models
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
face_recognition_model = dlib.face_recognition_model_v1(FACE_RECOGNITION_MODEL_PATH)

# Directory of known faces
KNOWN_FACES_DIR = "../data/known_faces"
OUTPUT_FILE = "../data/known_faces.pkl"

# Check if the known_faces directory exists
if not os.path.exists(KNOWN_FACES_DIR):
    print(f"Directory not found: {KNOWN_FACES_DIR}")
    print("Please create the directory and add known face images.")
    exit()

known_faces = {}

for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(KNOWN_FACES_DIR, filename)
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect face
        detected_faces = face_detector(gray_image)
        if len(detected_faces) != 1:
            print(f"Skipping {filename}: Found {len(detected_faces)} faces.")
            continue

        # Get the first detected face and compute descriptor
        shape = shape_predictor(gray_image, detected_faces[0])
        descriptor = face_recognition_model.compute_face_descriptor(image, shape)
        known_faces[filename.split(".")[0]] = descriptor

# Save known faces
with open(OUTPUT_FILE, "wb") as file:
    pickle.dump(known_faces, file)

print(f"Known faces database created successfully at {OUTPUT_FILE}.")
