import sys
sys.path.insert(0, 'python')
import cv2
import model
import util
from hand import Hand
from body import Body
import matplotlib.pyplot as plt
import copy
import numpy as np
import scipy.misc
from Hand_Detection import detect_hand
from PIL import Image
from load_image import read_image
import random

def parsearg():
    if len(sys.argv) > 1:
        if sys.argv[1] == 'reload':
            return True
    return False

flag = parsearg()
print(flag)
if flag:
    imglist = read_image(['earring_modified', 'necklace_modified'])

body_estimation = Body('model/body_pose_model.pth')
hand_estimation = Hand('model/hand_pose_model.pth')
bias = 20
#bias2 = 5
cap = cv2.VideoCapture('video/1559875195004305.mp4')
ret, oriImg = cap.read()
vid_writer = cv2.VideoWriter('paohui.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (oriImg.shape[1],oriImg.shape[0]))
#print(oriImg.shape[1],oriImg.shape[0])
img2 = Image.open('cutouts/earring/earring0.png')
img3 = Image.open('images/necklace2.jpg')
img2 = scipy.misc.imresize(img2, (70, 30))
img3 = scipy.misc.imresize(img3, (250, 300))
#print(img2.shape)
rows,cols,channels = img2.shape
rows3,cols3,channels3 = img3.shape
cap.set(3, 640)
cap.set(4, 480)
count = 0

while True:
    ret, oriImg = cap.read()
    if not ret:
        break
    
    if flag and count%15 == 0:
        earring = random.randint(0, len(imglist[0]) - 1)
        neck = random.randint(0, len(imglist[1]) - 1)
        img2 = imglist[0][earring]
        img3 = imglist[1][neck]
        img2 = scipy.misc.imresize(img2, (70, 30))
        img3 = scipy.misc.imresize(img3, (250, 300))
        rows,cols,channels = img2.shape
        rows3,cols3,channels3 = img3.shape
    
    candidate, subset = body_estimation(oriImg)
    print(candidate, subset)
    '''
    if len(candidate) != len(subset[0]):
        for i in range(len(subset[0])-2):
            if subset[0][i] != -1 and (candidate[int(subset[0][i])][0] == oriImg.shape[1]-1 or candidate[int(subset[0][i])][1] == oriImg.shape[0]-1):
                subset[0][i] = -1
    '''
    canvas = copy.deepcopy(oriImg)
    #canvas = Image.fromarray(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB))
    #canvas = canvas.transpose(Image.FLIP_LEFT_RIGHT)
    #canvas = util.draw_bodypose(canvas, candidate, subset)
    
    point = candidate[int(subset[0][16])]
    point2 = candidate[int(subset[0][17])]
    point3 = candidate[int(subset[0][1])]
    #img2 = np.array(img2)
    #print(img2.shape)
    #print(img2[0].shape)
    
    
    roi = canvas[int(point[1] + bias) : int(point[1] + rows + bias), int(point[0] - cols/2) : int(point[0] + cols/2)]
    roi2 = canvas[int(point2[1] + bias) : int(point2[1] + rows + bias), int(point2[0] - cols/2) : int(point2[0] + cols/2)]
    roi3 = canvas[int(point3[1] - 2*rows3/3) : int(point3[1] + rows3/3), int(point3[0] - cols3/2) : int(point3[0] + cols3/2)]
    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 200, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    img3gray = cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
    ret3, mask3 = cv2.threshold(img3gray, 200, 255, cv2.THRESH_BINARY)
    mask_inv3 = cv2.bitwise_not(mask3)
    # Now black-out the area of logo in ROI
    print(roi2.shape, mask.shape)
    if roi.shape[0] == mask.shape[0] and roi.shape[1] == mask.shape[1]:
        img1_bg = cv2.bitwise_and(roi,roi,mask = np.uint8(mask))
    if roi2.shape[0] == mask.shape[0] and roi2.shape[1] == mask.shape[1]:
        img1_bg2 = cv2.bitwise_and(roi2,roi2,mask = np.uint8(mask))
    if roi3.shape[0] == mask3.shape[0] and roi3.shape[1] == mask3.shape[1]:
        img1_bg3 = cv2.bitwise_and(roi3,roi3,mask = np.uint8(mask3))
    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2,img2,mask = np.uint8(mask_inv))
    img3_fg = cv2.bitwise_and(img3,img3,mask = np.uint8(mask_inv3))
    #print(img1_bg.shape)
    #print(img2_fg[:][:][:3].shape)
    # Put logo in ROI and modify the main image
    #print(type(img2_fg))
    
    if roi.shape[0] == mask.shape[0] and roi.shape[1] == mask.shape[1]:
        dst = cv2.add(img1_bg, img2_fg[:, :, (2, 1, 0)])
    if roi2.shape[0] == mask.shape[0] and roi2.shape[1] == mask.shape[1]:
        dst2 = cv2.add(img1_bg2, img2_fg[:, :, (2, 1, 0)])
    if roi3.shape[0] == mask3.shape[0] and roi3.shape[1] == mask3.shape[1]:
        dst3 = cv2.add(img1_bg3, img3_fg[:, :, (2, 1, 0)])
    
    if roi.shape[0] == mask.shape[0] and roi.shape[1] == mask.shape[1]:
        canvas[int(point[1] + bias) : int(point[1] + rows + bias), int(point[0] - cols/2) : int(point[0] + cols/2)] = dst
    if roi2.shape[0] == mask.shape[0] and roi2.shape[1] == mask.shape[1]:
        canvas[int(point2[1] + bias) : int(point2[1] + rows + bias), int(point2[0] - cols/2) : int(point2[0] + cols/2)] = dst2
    if roi3.shape[0] == mask3.shape[0] and roi3.shape[1] == mask3.shape[1]:
        canvas[int(point3[1] - 2*rows3/3) : int(point3[1] + rows3/3), int(point3[0] - cols3/2) : int(point3[0] + cols3/2)] = dst3
    '''
    hands_list = util.handDetect(candidate, subset, oriImg)

    all_hand_peaks = []
    for x, y, w, is_left in hands_list:
        peaks = hand_estimation(oriImg[y:y+w, x:x+w, :])
        peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
        peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
        all_hand_peaks.append(peaks)

    canvas = util.draw_handpose(canvas, all_hand_peaks)
    '''
    cv2.imshow('demo', canvas)#一个窗口用以显示原视频
    
    key = cv2.waitKey(1)
    if key == 27:
        break
    vid_writer.write(canvas)
    count += 1

vid_writer.release()
cv2.destroyAllWindows()
