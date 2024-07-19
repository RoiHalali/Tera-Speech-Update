"Name       : Roi Halali & Dor Kershberg "
"Titel      : edge and color detect      "

import cv2 
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
import imutils
import mediapipe as mp
# from face_landmark import *


NUM_FACE = 1
            
class FaceLandMarks():
    def __init__(self, staticMode=False,maxFace=NUM_FACE, minDetectionCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFace =  maxFace
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFace, self.minDetectionCon, self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def findFaceLandmark(self, img, draw=False):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)

        faces = []
        
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACE_CONNECTIONS, self.drawSpec, self.drawSpec)

                face = []
                for id, lm in enumerate(faceLms.landmark):
                    # print(lm)
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    # cv2.putText(img, str(id), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0,255,0), 1)
                    # print(id, x, y)
                    face.append([x,y])
                faces.append(face)
        return img, faces

def simple_edge_detection(image): 
   edges_detected = cv2.Canny(image , 100, 200) 
   images = [image , edges_detected]
   
def dist(x1,y1,x2,y2):
    distance= np.sqrt(np.square(x1-x2)+np.square(y1-y2))
    return distance 
   
def cropp_img(full_img,x,y,h,w):
    # Convert into grayscale
    # Draw rectangle around the faces and crop the faces
    cropped_face = full_img[(y-300):y + h+300, x:x + w]
    return(cropped_face)

def cv_print (img,windowname):
    #get RGB img  
    cv2.namedWindow(windowname,0)
    cv2.resizeWindow(windowname, 400,400)
    cv2.imshow(windowname, img)
    
    cv2.waitKey(0)    
    cv2.destroyAllWindows()

def Tongue_detect(img,tounge_state,state):

    if tounge_state=='down':
        x=int(state[0]*0.07)
        y=int(state[1]*0.1)
        
    if tounge_state=='up':
        x=int(state[0]*0.1)
        y=int(state[1]*0.08)
        
    if tounge_state=='left' or tounge_state=='right':
        x=int(state[0]*0.11)
        y=int(state[1]*0.08)

    #cv_print (blur,"blur")
    blur = cv2.blur(img, (20,20))   #blur for decrease noise

    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    img_filter = cv2.filter2D(blur, -1, kernel)

    hsv = cv2.cvtColor(img_filter, cv2.COLOR_RGB2HSV) #BGR to hsv
    
    #taking only red elements:
    mask_type=np.load("mask/mask_type.npy")
    mask_tune=np.load("mask/mask_tune.npy")

    if mask_type is not None:
        mask = mask_tune
    else :
        redlow = np.array([150, 0, 0])
        redup = np.array([180, 254, 254])
        mask = cv2.inRange(hsv, redlow, redup)
        # cv_print (mask,"mask")
    result=cv2.bitwise_and(img,img, mask = mask ) #BGR pic afer mask
    #cv_print (result,"result")
    gray=cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) #from BGR to gray
    #cv_print (gray,"gray")
  
    #elipses detection
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (x, y))*255
    elipse = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel,iterations=1)
    #cv_print (elipse,"elips")
    
    edges = cv2.Canny(elipse,75,170) # Canny Edge Detection 
    # cv_print (edges,"edges")
    
    return edges,gray

def is_contour_bad(c):
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	# the contour is 'bad' if it is not a rectangle
	return not len(approx) == 4

def cleaning(img,mini):
        
        cnts = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            area = cv2.contourArea(c)
            if area <mini :
                cv2.drawContours(img, [c], -1, (0,0,0), -1)
        
        # Morph close and invert image
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        close =  cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        return close                

def cropped_face(img):
    
    # img=cv2.resize(img,(400,400))
    
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    
    # Load the cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')    
  
    # Detect all faces
    faces = face_cascade.detectMultiScale(gray, 1.35, 1)    
    if len(faces)!=0:
        wt,ht=0,0   # initial val for rect size
        for (x, y, w, h) in faces:
            # find the biggest rect:
            if (w+h>=wt+ht):
                wt,ht=w,h
                faces_rec = img[y+25:y + h+45, x:x + w]
                # Draw rectangle around the faces and crop the faces
                #cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 150), 2)
                #plt.imshow(faces_rec)
        if faces_rec.shape[0]>10:
        # printing the final rectangular image
        #cv_print(faces_rec,"rect on img") 
            return (faces_rec) 
        else: return []
    else: return []

def print_dots(img,edge):
    
    indices=np.where(edge!=[0])
    cv2.imshow('img1',img)
    circled_pic = cv2.circle(img, indices, radius=3, color=(100, 0, 100),
                             thickness=-1)  # draw points
    plt.imshow(img)
    cv2.circle(img, (indices[1],indices[0]), radius=0, color=(0, 0, 255), thickness=-1)
    plt.scatter( indices[1] , indices[0],s=1 , c='yellow',vmin=0, vmax=255)
    plt.show()
    return circled_pic
    
def tounge_down(img, factor, tounge_state):
    # live cam analysis:
    cropp_flag = False
    while cropp_flag == False:
        cropp_img = cropped_face(img)
        if len (cropp_img) != 0:
            cropp_flag = True
            cropp_img = cv2.resize(cropp_img, (428, 354))
            state=[cropp_img.shape[0],cropp_img.shape[1]]
            tongue_edge,tongue_gray=Tongue_detect(cropp_img, tounge_state, state)
            mini=tongue_edge.shape[0]*factor
            cleaned = tongue_edge
            cleaned = cleaning(tongue_edge,mini)
            cleaned = morphology.remove_small_objects(tongue_edge, min_size=mini, connectivity=8)
           
            if (np.argmax(cleaned)!=0):
                x, y, w, h = cv2.boundingRect(cleaned)
                if tounge_state =='right':
                    loc = (x, np.argmax(cleaned[:, x]))
                elif tounge_state =='left':
                    loc = (x+w-1, np.argmax(cleaned[:, x+w-1]))   #up 
                elif tounge_state =='up':
                    loc = (np.argmax(cleaned[y, :]), y)   
                elif tounge_state =='down':
                    loc = (np.argmax(cleaned[y+h-1, :]), y+h-1)   #down
            else:loc=[0,0]
            return cropp_img, cleaned, loc    

        else: return img,[],[] 
        
    
# framewidth=1912     
# framehight=1072  
# cap=cv2.VideoCapture(0)
# cap.set(3,framewidth)
# cap.set(4,framehight)
detector = FaceLandMarks()

def tounge(img, tounge_state):

    faces = None
    factor=0.85        
    cropp, cleaned, loc = tounge_down(img, factor, tounge_state)
    _, faces = detector.findFaceLandmark(cropp)        
    if faces != []:
        points = np.array(faces)
        points = np.reshape(points,[468,2])
       
        num_labels = 0
        if cleaned != [] :
            output = cv2.connectedComponentsWithStats(cleaned, 8, cv2.CV_32S)   #clear unnessery elements from edge
            num_labels = output[0]
        
        
        if loc != [] and loc != [0,0] and points.any() != None and num_labels == 2 :
            
            if (tounge_state == 'down'):
                distance = dist(points[16,0],points[16,1],loc[0],loc[1])
                ver_dis_pnt = cv2.circle(cropp, (loc[0], loc[1]), radius=3, color=(255, 255, 255),
                                       thickness=-1)  # draw points
                # cv2.imshow('cropp', cropp)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # print(distance)
                # ver_dis_pnt = cv2.circle(ver_dis_pnt, (points[11, 0], points[11, 1]), radius=3, color=(255, 255, 255),
                #                           thickness=-1)  # draw points
                # ver_dis_pnt = cv2.circle(ver_dis_pnt, (points[16, 0], points[16, 1]), radius=3, color=(255, 255, 255),
                #                           thickness=-1)  # draw points
                # hor_dis_pnt = cv2.circle(ver_dis_pnt, (points[61, 0], points[61, 1]), radius=3, color=(255, 255, 255),
                #                           thickness=-1)  # draw points
                # hor_dis_pnt = cv2.circle(hor_dis_pnt, (points[291, 0], points[291, 1]), radius=3, color=(255, 255, 255),
                #                           thickness=-1)  # draw points
                return distance

            elif (tounge_state == 'up'):
                distance = dist(points[11,0],points[11,1],loc[0],loc[1])
                return distance

            elif (tounge_state == 'right'):  #right side
                ver_dis_pnt = cv2.circle(cropp, (loc[0], loc[1]), radius=3, color=(255, 255, 255),
                                       thickness=-1)
                cv2.imshow('cropp', cropp)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                distance = dist(points[61,0],points[61,1],loc[0],loc[1])
                return distance

            elif (tounge_state == 'left'):  #left side
                distance = dist(points[291,0],points[291,1],loc[0],loc[1])
                return distance

        else: return 0
    

    else: return 0

 







