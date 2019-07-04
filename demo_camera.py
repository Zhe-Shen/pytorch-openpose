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

body_estimation = Body('model/body_pose_model.pth')
hand_estimation = Hand('model/hand_pose_model.pth')
bias = 20
#bias2 = 5
cap = cv2.VideoCapture('video/lhyhand.mp4')
ret, oriImg = cap.read()
vid_writer = cv2.VideoWriter('lhy.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (oriImg.shape[1],oriImg.shape[0]))
print(oriImg.shape[1],oriImg.shape[0])
#img2 = Image.open('images/earring.jpg')
#img3 = Image.open('images/necklace2.jpg')
img4 = Image.open('images/ring2.jpg')
#img2 = scipy.misc.imresize(img2, (70, 30))
#img3 = scipy.misc.imresize(img3, (250, 300))
img4 = scipy.misc.imresize(img4, (60, 60))
#print(img2.shape)
#rows,cols,channels = img2.shape
#rows3,cols3,channels3 = img3.shape
#rows,cols,channels = img4.shape
#cap.set(3, 640)
#cap.set(4, 480)

def Angle(v1,v2):
    dot = np.dot(v1,v2)
    x_modulus = np.sqrt((v1*v1).sum())
    y_modulus = np.sqrt((v2*v2).sum())
    cos_angle = dot / x_modulus / y_modulus
    angle = np.degrees(np.arccos(cos_angle))
    return angle

def rotate_image(img, angle, crop):
    """
    angle: 旋转的角度
    crop: 是否需要进行裁剪，布尔向量
    """
    w, h = img.shape[:2]
    # 旋转角度的周期是360°
    angle %= 360
    # 计算仿射变换矩阵
    M_rotation = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    # 得到旋转后的图像
    img_rotated = cv2.warpAffine(img, M_rotation, (w, h))

    # 如果需要去除黑边
    if crop:
        # 裁剪角度的等效周期是180°
        angle_crop = angle % 180
        if angle > 90:
            angle_crop = 180 - angle_crop
        # 转化角度为弧度
        theta = angle_crop * np.pi / 180
        # 计算高宽比
        hw_ratio = float(h) / float(w)
        # 计算裁剪边长系数的分子项
        tan_theta = np.tan(theta)
        numerator = np.cos(theta) + np.sin(theta) * np.tan(theta)

        # 计算分母中和高宽比相关的项
        r = hw_ratio if h > w else 1 / hw_ratio
        # 计算分母项
        denominator = r * tan_theta + 1
        # 最终的边长系数
        crop_mult = numerator / denominator

        # 得到裁剪区域
        w_crop = int(crop_mult * w)
        h_crop = int(crop_mult * h)
        x0 = int((w - w_crop) / 2)
        y0 = int((h - h_crop) / 2)

        img_rotated = crop_image(img_rotated, x0, y0, w_crop, h_crop)

    return img_rotated


while True:
    ret, oriImg = cap.read()
    if not ret:
        break
    oriImg = Image.fromarray(cv2.cvtColor(oriImg,cv2.COLOR_BGR2RGB))
    oriImg = oriImg.transpose(Image.FLIP_LEFT_RIGHT)
    oriImg = cv2.cvtColor(np.asarray(oriImg),cv2.COLOR_RGB2BGR)
    #candidate, subset = body_estimation(oriImg)
    #print(candidate, subset)
    '''
    if len(candidate) != len(subset[0]):
        for i in range(len(subset[0])-2):
            if subset[0][i] != -1 and (candidate[int(subset[0][i])][0] == oriImg.shape[1]-1 or candidate[int(subset[0][i])][1] == oriImg.shape[0]-1):
                subset[0][i] = -1
    '''
    canvas = copy.deepcopy(oriImg)
    ring = copy.deepcopy(img4)
    #canvas = Image.fromarray(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB))
    #canvas = canvas.transpose(Image.FLIP_LEFT_RIGHT)
    #canvas = util.draw_bodypose(canvas, candidate, subset)
    '''
    point = candidate[int(subset[0][16])]
    point2 = candidate[int(subset[0][17])]
    point3 = candidate[int(subset[0][1])]
    #img2 = np.array(img2)
    #print(img2.shape)
    #print(img2[0].shape)
    
    
    roi = canvas[int(point[1] + bias) : int(point[1] + rows + bias), int(point[0] - cols/2) : int(point[0] + cols/2)]
    roi2 = canvas[int(point2[1] + bias) : int(point2[1] + rows + bias), int(point2[0] - cols/2) : int(point2[0] + cols/2)]
    roi3 = canvas[int(point3[1] - rows3/2) : int(point3[1] + rows3/2), int(point3[0] - cols3/2) : int(point3[0] + cols3/2)]
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
        canvas[int(point3[1] - rows3/2) : int(point3[1] + rows3/2), int(point3[0] - cols3/2) : int(point3[0] + cols3/2)] = dst3
    '''
    # detect hand
    #hands_list = util.handDetect(candidate, subset, oriImg)
    #canvas = cv2.cvtColor(np.asarray(canvas),cv2.COLOR_RGB2BGR)
    hands_list = detect_hand(canvas)
    #print(hands_list)
    all_hand_peaks = []
    for x, y, w, is_left in hands_list:
        peaks = hand_estimation(oriImg[y:y+w, x:x+w, :])
        peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
        peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
        all_hand_peaks.append(peaks)
    print(all_hand_peaks)
    p1 = all_hand_peaks[0][13]
    p2 = all_hand_peaks[0][14]
    if not ((p1[0] == 0 and p1[1] == 0) or (p2[0] == 0 and p2[1] == 0)):
        degree = Angle(p1 - p2, np.array([1, 0]))
        print(degree)
        crop_image = lambda ring, x0, y0, w, h: ring[x0:x0+w, y0:y0+h]
        ring = rotate_image(ring, 90 - degree, True)
        #ring = Image.fromarray(cv2.cvtColor(ring,cv2.COLOR_BGR2RGB))
        #ring = ring.rotate(90 - degree)
        #ring = cv2.cvtColor(np.asarray(ring),cv2.COLOR_RGB2BGR)
        #ring = clear_black(ring)
        rows,cols,channels = ring.shape

        p = p1/3 + 2*p2/3
        roi = canvas[int(p[1] - rows/2): int(p[1] + rows/2), int(p[0] - cols/2): int(p[0] + cols/2)]

        img2gray = cv2.cvtColor(ring,cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 200, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        img1_bg = cv2.bitwise_and(roi,roi,mask = np.uint8(mask))
        img2_fg = cv2.bitwise_and(ring,ring,mask = np.uint8(mask_inv))
        dst = cv2.add(img1_bg, img2_fg[:, :, (2, 1, 0)])
        canvas[int(p[1] - rows/2): int(p[1] + rows/2), int(p[0] - cols/2): int(p[0] + cols/2)] = dst
        #canvas = util.draw_handpose(canvas, all_hand_peaks)
    
    canvas = Image.fromarray(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB))
    canvas = canvas.transpose(Image.FLIP_LEFT_RIGHT)
    canvas = cv2.cvtColor(np.asarray(canvas),cv2.COLOR_RGB2BGR)
    
    cv2.imshow('demo', canvas)#一个窗口用以显示原视频
    
    key = cv2.waitKey(1)
    if key == 27:
        break
    vid_writer.write(canvas)

vid_writer.release()
cv2.destroyAllWindows()

