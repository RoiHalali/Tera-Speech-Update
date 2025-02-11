"Name       : Roi Halali & Dor Kershberg "
"Titel      : edge and color detect      "


import cv2
import numpy as np
import mediapipe as mp
import timeit
import matplotlib.pyplot as plt
from skimage import morphology
from scipy.spatial import distance
from calibrate import *
from opencvcoloredge import *

NUM_FACE = 1
            
def dist(x1,y1,x2,y2):
    distance= np.sqrt(np.square(x1-x2)+np.square(y1-y2))
    return distance 

def MAR_lips (xru,yru,xrd,yrd,xmu,ymu,xmd,ymd,xlu,ylu,xld,yld,xvl,yvl,xvr,yvr): #r- right l- left u-up d-down v-verticel
    A = dist(xru,yru,xrd,yrd)
    B = dist(xmu,ymu,xmd,ymd)
    C = dist(xlu,ylu,xld,yld)
    L = (A+B+C)/3 
    D = dist(xvl,yvl,xvr,yvr) 
    mar=np.exp(L/D+10)
    return mar

class FaceLandMarks():
    def __init__(self, staticMode=False,maxFace=NUM_FACE, minDetectionCon=0.5, minTrackCon=0.1):
        self.staticMode = staticMode
        self.maxFace =  maxFace
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFace, self.minDetectionCon, self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def findFaceLandmark(self, img, draw= False):
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

def cropp_img(full_img,x,y,h,w):
    # Convert into grayscale
    # Draw rectangle around the faces and crop the faces
    cropped_face = full_img[(y-300):y + h+300, x:x + w]
    return cropped_face

def vertical_horizontal(img):
        # Canera setings: #TODO: finde otomaticly the best camera options.
        # framewidth = 1912
        # framehight = 1072
        # cap = cv2.VideoCapture(0)
        # # cap.set(3, framewidth)
        # # cap.set(4, framehight)
        # img = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB)
        face_width = 0
        # Face-Landmark setings:
        detector = FaceLandMarks()
        img, faces = detector.findFaceLandmark(img)

        
    
        # Face indentify validation:
        if faces != []:
            points = np.array(faces)
            if points.shape[0] == 0:
                img = cv2.resize(img, (428, 354))  # defult size
                return 0, 0
                
            else:   
                    points = np.reshape(points, [468, 2])
                    # Evaluate vertical and  horizontal distance:
                    d = distance.euclidean([points[11, 0], points[11, 1]], [points[16, 0], points[16, 1]])
                    # vertical_distans = dist(points[11, 0], points[11, 1], points[16, 0], points[16, 1])  # distance
                    # ver_dis_pnt = cv2.circle(img, (points[11, 0], points[11, 1]), radius=3, color=(255, 255, 255),
                    #                           thickness=-1)  # draw points
                    # ver_dis_pnt = cv2.circle(ver_dis_pnt, (points[16, 0], points[16, 1]), radius=3, color=(255, 255, 255),
                    #                           thickness=-1)  # draw points
                    # cv2.imshow('image', cv2.resize(ver_dis_pnt, (300, 300)))#plot image with points
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
            
                    horizontal_distans = dist(points[61, 0], points[61, 1], points[291, 0], points[291, 1])  # distance
                    # hor_dis_pnt = cv2.circle(ver_dis_pnt, (points[61, 0], points[61, 1]), radius=3, color=(255, 255, 255),
                    #                           thickness=-1)  # draw points
                    # hor_dis_pnt = cv2.circle(hor_dis_pnt, (points[291, 0], points[291, 1]), radius=3, color=(255, 255, 255),
                    #                           thickness=-1)  # draw points
                    # cv2.imshow('image', cv2.resize(hor_dis_pnt, (300, 300))) #plot image with points
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
            
                    ## Calculate Frame per second:
                    # cTime = time.time()
                    # fps = 1 / (cTime - pTime)
                    # pTime = cTime
            
                    # img = cv2.resize(img, (428, 354)) #defult size

                    return d, horizontal_distans
        return 0, 0

def face(img, vertical_distans,horizontal_distans, request, reps):
    
    # print("ver= "+ str(vertical_distans))
    # print("hor= "+str(horizontal_distans))
    
    if  vertical_distans > 30 and request=="open_mouth" :
        return "open", vertical_distans, 0
        
    elif vertical_distans < 20 and request=="open_mouth" :
        return "close", 0, 0
    
    elif horizontal_distans > 70 and vertical_distans < 35 and request=="smile" :
        return "open", horizontal_distans, 0
    
    elif horizontal_distans <60 and  request=="smile" :
        return "close", 0, 0
        
    elif horizontal_distans < 55  and request=="kiss" :
        return "open", horizontal_distans, 0
    
    elif horizontal_distans > 60  and request=="kiss" :
        return "close", 0, 0
    
    elif request == "down":
        
        distance = tounge(img, request)
        # print("distance: ", distance)
        
        if vertical_distans < 30 and distance < 30  :
            return "close", 0, 0
        
        elif distance != 0:
            if distance > 30 and distance < 90 :
                reps += 1
                if reps == 3:
                    print ("distance: ", distance)
                    return "open", distance, reps
                else: return "open", 0 ,reps
            else: return 'close', 0, reps
        
    elif request == "up":
        
        distance = tounge(img, request)
        # print("distance: ", distance)
        
        if vertical_distans < 30 and distance < 30  :
            return "close", 0, 0
        
        elif distance != 0:
            if distance > 30 and distance < 90 :
                reps += 1
                if reps == 3:
                    print ("distance: ", distance)
                    return "open", distance, reps
                else: return "open", 0 ,reps
            else: return 'close', 0, reps
        
    elif request == "left":
        
        distance = tounge(img, request)
        # print("distance: ", distance)
        
        if vertical_distans < 30 and distance < 30  :
            return "close", 0, 0
        
        elif distance != 0:
            if distance > 30 and distance < 90 :
                reps += 1
                if reps == 3:
                    print ("distance: ", distance)
                    return "open", distance, reps
                else: return "open", 0 ,reps
            else: return 'close', 0, reps
        
    elif request == "right":
        
        distance = tounge(img, request)
        # print("distance: ", distance)
        
        if vertical_distans < 30 and distance < 30  :
            return "close", 0, 0
        
        elif distance != 0:
            if distance > 30 and distance < 90 :
                reps += 1
                if reps == 3:
                    print ("distance: ", distance)
                    return "open", distance, reps
                else: return "open", 0 ,reps
            else: return 'close', 0, reps
            
    return 'close', 0, 0

        


