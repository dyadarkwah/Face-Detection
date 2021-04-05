from cv2 import cv2
import itertools

# Load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
trained_face_data_alt = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
# trained_full_body = cv2.CascadeClassifier('haarcascade_fullbody.xml')



# Capture video from webcam. 0 is default cam
webcam = cv2.VideoCapture(0)



# Iterate forever over frames
while True:

    ### Read the current frame. read returns bool, frame
    successful_frame_read, frame = webcam.read()

    ## Converting to grayscale
    grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ## Detect faces, also of different sizes, and returns coordinates of face in rectangle [[x1, y1] [w, h])
    face_coordinates = trained_face_data.detectMultiScale(frame)
    face_coordinates_alt = trained_face_data_alt.detectMultiScale(frame)
    # body_coordinates = trained_full_body.detectMultiScale(frame)

    ## Draw rectangles around the faces. Color in BGR
    for ((x, y, w, h),(x, y, w, h)) in zip(face_coordinates, face_coordinates_alt):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
    
    # for (x, y, w, h) in body_coordinates:
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)


    ## Display image with faces
    cv2.imshow('Face Detector', frame)
    key = cv2.waitKey(1)

    #Stop if 'Q' is pressed
    if key == 81 or key == 113:
        break


## Choose an image to dectect faces
# img = cv2.imread('RDJ.png')
# faces = cv2.imread('faces.jpg')

print("Code Completed")