
import cv2
import imutils
import dlib
from imutils import face_utils
from scipy.spatial import distance
from pygame import mixer

# Initialize pygame mixer for sound alert
mixer.init()
mixer.music.load("music.wav")

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Constants for drowsiness detection
thresh = 0.25
frame_check = 20

# Initialize dlib's face detector and shape predictor
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Get the indexes of the facial landmarks for the left and right eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# Start video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video device")
    exit()

flag = 0
no_face_detected_flag = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture frame from webcam")
        break
    
    frame = imutils.resize(frame, width=450)
    
    # Ensure frame is in BGR format
    if frame.ndim != 3 or frame.shape[2] != 3:
        print("Error: Frame is not in BGR format")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    subjects = detect(gray, 0)
    
    if len(subjects) == 0:
        no_face_detected_flag += 1
        if no_face_detected_flag >= frame_check:
            cv2.putText(frame, "LOOK AT THE ROAD!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            mixer.music.play()
    else:
        no_face_detected_flag = 0
        for subject in subjects:
            shape = predict(gray, subject)
            shape = face_utils.shape_to_np(shape)
            
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            
            ear = (leftEAR + rightEAR) / 2.0
            
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            
            # Draw a rectangle around the face
            (x, y, w, h) = face_utils.rect_to_bb(subject)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            if ear < thresh:
                flag += 1
                
                if flag >= frame_check:
                    cv2.putText(frame, "DROWSY!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    mixer.music.play()
                else:
                    cv2.putText(frame, "Eyes Closed", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                flag = 0
                cv2.putText(frame, "Eyes Open", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display additional information
    cv2.putText(frame, "Drowsiness Detection System", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Press 'X' to exit", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display the resulting frame
    cv2.imshow("Drowsiness Detection", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("x"):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
