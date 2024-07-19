import cv2
from tkinter import *




def check_if_good(img, face_width_in_frame):
    # distance from camera to object(face) measured
    # centimeter
    Known_distance = 37
    fonts = cv2.FONT_HERSHEY_COMPLEX

    # width of face in the real world or Object Plane
    # centimeter
    Known_width = 20
     
    # find the face width(pixels) in the reference_image
    ref_image_face_width = 278
     
    # get the focal by calling "Focal_Length_Finder"
    # face width in reference(pixels),
    # Known_distance(centimeters),
    # known_width(centimeters)
    Focal_length_found = Focal_Length_Finder(
        Known_distance, Known_width, ref_image_face_width)
     
     
     # calling face_data function to find
    # the width of face(pixels) in the frame     
    # check if the face is zero then not
    # find the distance
    if face_width_in_frame != 0:
       
        # finding the distance by calling function
        # Distance finder function need
        # these arguments the Focal_Length,
        # Known_width(centimeters),
        # and Known_distance(centimeters)
        Distance = Distance_finder(
            Focal_length_found, Known_width, face_width_in_frame)
                # draw line as background of text
        # cv2.line(img, (30, 30), (230, 30), (0, 0, 255), 32)
        # cv2.line(img, (30, 30), (230, 30), (0, 0, 0), 28)
     
        if (Distance > 45 and Distance < 70):
            # Drawing Text on the screen
            # cv2.putText(
            #     img, f"Good distance: {round(Distance,2)} CM", (30, 35), fonts,
            #   0.6, (0, 0, 0), 2)
            return 'good'
        elif Distance < 45:
            # cv2.putText(
            #     img, f"Move back: {round(Distance,2)} CM", (30, 35), fonts,
            #   0.6, (0, 0, 0), 2)
            return 'too_close'

        elif Distance > 60:
            # cv2.putText(
            #     img, f"Move forward: {round(Distance,2)} CM", (30, 35), fonts,
            #   0.6, (0, 0, 0), 2)
            return "too_far"

    
def face_data(image):
    face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    face_width = 0  # making face width to zero
 
    # converting color image to gray scale image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
    # detecting face in the image
    faces = face_detector.detectMultiScale(gray_image, 1.3, 5)
 
    # looping through the faces detect in the
    # image getting coordinates x, y ,
    # width and height
    for (x, y, h, w) in faces:
  
        # getting face width in the pixels
        face_width = w
 
    # return the face width in pixel
    return face_width

# focal length finder function
def Focal_Length_Finder(measured_distance, real_width, width_in_rf_image):
 
    # finding the focal length
    focal_length = (width_in_rf_image * measured_distance) / real_width
    return focal_length
 
# distance estimation function
def Distance_finder(Focal_Length, real_face_width, face_width_in_frame):
 
    distance = (real_face_width * Focal_Length)/face_width_in_frame
    # return the distance
    return distance


 

