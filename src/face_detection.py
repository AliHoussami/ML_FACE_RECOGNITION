import dlib
import cv2


class FaceDetector:
    def __init__(self, shape_predictor_path, face_recognition_model_path):
        # Initialize Dlib models
        self.face_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(shape_predictor_path)
        self.face_recognition_model = dlib.face_recognition_model_v1(face_recognition_model_path)

    def detect_faces(self, image):
        """Detect faces and return bounding boxes and descriptors."""
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detected_faces = self.face_detector(gray_image)
        faces = []

        for face_rect in detected_faces:
            # Get face bounding box
            bbox = {
                "left": face_rect.left(),
                "top": face_rect.top(),
                "right": face_rect.right(),
                "bottom": face_rect.bottom()
            }

            # Get facial landmarks
            shape = self.shape_predictor(gray_image, face_rect)
            landmarks = [(point.x, point.y) for point in shape.parts()]

            # Compute face descriptor
            face_descriptor = self.face_recognition_model.compute_face_descriptor(image, shape)

            faces.append({
                "bounding_box": bbox,
                "landmarks": landmarks,
                "descriptor": face_descriptor
            })

        return faces

    @staticmethod
    def draw_faces(image, faces):
        """Draw bounding boxes and landmarks on the image."""
        for face in faces:
            # Draw bounding box
            bbox = face["bounding_box"]
            cv2.rectangle(image, (bbox["left"], bbox["top"]), (bbox["right"], bbox["bottom"]), (0, 255, 0), 2)

            # Draw landmarks
            for (x, y) in face["landmarks"]:
                cv2.circle(image, (x, y), 2, (0, 0, 255), -1)

        return image
