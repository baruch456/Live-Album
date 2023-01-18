import cv2
import numpy as np
import multiprocessing 
import ffpyplayer.player as ffp
import pygame
from videosClass import videosProp
from moviepy.editor import *
from threading import Thread

def checkFound(des1,des2):
    FLANN_INDEX_KDTREE = 6
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict() # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    good1=[]
    for i, pair in enumerate(matches):
        try:
            m, n = pair
            if m.distance<0.75 *n.distance:
                good1.append(m)
            # print(len(good1))
        except ValueError:
            pass
    if len(good1)>33:
        # print('find')
        return good1
    else:
        return None


if __name__ == "__main__":

    pathVid=r"C:\Users\avimo\OneDrive\Desktop\OpenCv\projects\ArAlbum\data_Vidoes"
    pathSound=r"C:\Users\avimo\OneDrive\Desktop\OpenCv\projects\ArAlbum\data_Sound"
    pathImg=r"C:\Users\avimo\OneDrive\Desktop\OpenCv\projects\ArAlbum\data_image"

    
    firstDetect= False

    imgDatabase = next(os.walk("data_image"))[2]
    vidDatabase = next(os.walk("data_Vidoes"))[2]
    imageTargetDatabase=[]
    myVidDatabase=[]
    for myVidN in vidDatabase:
        name=r"data_Vidoes\\{}".format(myVidN)
        myVid = cv2.VideoCapture(name)
        myVidDatabase.append(myVid)

    soundDatabase = next(os.walk("data_Sound"))[2]
    orbsDatabase=[]
    orb = cv2.ORB_create()

    for img in imgDatabase:
        imgname=r"data_image\\{}".format(img)
        imgTarget= cv2.imread(imgname)
        imageTargetDatabase.append(imgTarget)
        kp1, des1 = orb.detectAndCompute(imgTarget,None)
        orbs1=[kp1,des1]
        orbsDatabase.append(orbs1)

    framesCounter=0
    cap = cv2.VideoCapture(0)
    img2=np.zeros((10,10,3), dtype=np.uint8)
    imgAug=np.zeros((10,10,3), dtype=np.uint8)
    imgWarp=np.zeros((10,10,3), dtype=np.uint8)



    lastFoundKey=None
    while True:
        sucess,imgwebcome=cap.read()
        imgAug=imgwebcome.copy()


        orb = cv2.ORB_create()
        kp2, des2 = orb.detectAndCompute(imgwebcome, None)
        detect= False

        for count, value in enumerate(orbsDatabase):
            good=checkFound(value[1],des2)
            if good is not None:
                keyDetect=count
                detect=True
                if firstDetect is False:
                    firstDetect=True
                break

        if detect:
            if lastFoundKey is not None and lastFoundKey != keyDetect :
                framesCounter=0

            imgTarget=imageTargetDatabase[keyDetect]
            hT,wT,ct=imgTarget.shape
            song=soundDatabase[keyDetect]
            myVid=myVidDatabase[keyDetect]
            if framesCounter ==  int(myVid.get(cv2.CAP_PROP_FRAME_COUNT)):
                framesCounter=0

            if framesCounter==0:
                myVid.set(cv2.CAP_PROP_POS_FRAMES,0)

            success, imgVideo = myVid.read()
            imgVideo=cv2.resize(imgVideo,(wT,hT))


            
            #מוצא את מיקום נקודה בשני התמונות ויוצר מטריצת התאמה ויחסות בינהם 3 על 3
            kp1=orbsDatabase[keyDetect][0]
            srcPts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dstPts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            #הטלה של הנקודות בתמונה הראשונה בתמונה השנייה.
            matrix, mask = cv2.findHomography(srcPts,dstPts,cv2.RANSAC,5) 
            # print(matrix)
            # למצוא את המסגרת של התמונה המקורית הטרגט ימץ
            pts = np.float32([[0,0],[0,hT],[wT,hT],[wT,0]]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,matrix)
            #הנקודות שבהם הוא משער שהוא מצא את המסגרת
            img2 = cv2.polylines(imgwebcome,[np.int32(dst)],True,(255,0,255),3)
            # cv2.imshow('img2',img2)

            #התמונה הגזורה מהוידאו בפרספקטיבה של התמונת אלבום
            imgWarp = cv2.warpPerspective(imgVideo,matrix, (imgwebcome.shape[1],imgwebcome.shape[0]))

            #--------הלבשה של הוידאו על התמונה באלבום
            maskNew = np.zeros((imgwebcome.shape[0],imgwebcome.shape[1]),np.uint8)
            # ואז בשחור - צביעת האזור שבאלבום בלבן וכל השאר בשחור!
            cv2.fillPoly(maskNew,[np.int32(dst)],(255,255,255))
            maskInv = cv2.bitwise_not(maskNew)
            # לבישת שחור על הבסיס הלבן בלבד, בשילוב הרקע
            imgAug = cv2.bitwise_and(imgAug,imgAug,mask = maskInv)
            imgAug = cv2.bitwise_or(imgWarp,imgAug)
            framesCounter+=1
            lastFoundKey=keyDetect
        # else:
            # song.set_volume(0.0)

        cv2.imshow('imgAug',imgAug) 
        cv2.waitKey(1)
            