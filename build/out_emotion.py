import cv2
import dlib
import numpy as np
from imutils import face_utils
from sklearn.externals import joblib

# Load the face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load the machine learning model
clf = joblib.load("facial_expression_model.pkl")

# Define the list of emotion labels
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

# Load the test image and detect the face
test_image = cv2.imread("test_image.jpg")
gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
rects = detector(gray, 1)

# Loop over the detected faces
for (i, rect) in enumerate(rects):
    # Detect facial landmarks
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    
    # Extract features from the landmarks
    features = []
    for (i, (x, y)) in enumerate(shape):
        for (j, (x2, y2)) in enumerate(shape):
            if i < j:
                features.append(np.linalg.norm(np.array([x, y]) - np.array([x2, y2])))
    
    # Make a prediction using the trained model
    prediction = clf.predict(np.asarray(features).reshape(1, -1))
    label = EMOTIONS[prediction[0]]
    
    # Display the results
    cv2.rectangle(test_image, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 0), 2)
    cv2.putText(test_image, label, (rect.left(), rect.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    
cv2.imshow("Output", test_image)
cv2.waitKey(0)