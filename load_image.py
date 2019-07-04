import os
from PIL import Image


def read_image(paths):
    res = []
    for path in paths:
        imglist = []
        for filename in os.listdir(r'./cutouts/' + path):
            img = Image.open('cutouts/' + path + '/' + filename)
            imglist.append(img)
        res.append(imglist)
    return res
