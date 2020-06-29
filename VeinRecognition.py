import cv2
import numpy as np
import matplotlib.pyplot as plt
def GaussMaskCreate(mask_size,sigma):
    mask_size1=mask_size//2
    x=np.arange(0,mask_size)-mask_size1
    y=np.arange(0,mask_size)-mask_size1
    x, y = np.meshgrid(x, y)
    Mask = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)
    return Mask
imsi=cv2.imread("./1.png",cv2.IMREAD_GRAYSCALE)
imsi=cv2.resize(imsi,dsize=(400,400),interpolation=cv2.INTER_AREA)
LowpassFilterMask=GaussMaskCreate(3,10)
plt.imshow(imsi,'gray')
plt.show()
for i in range(5):
    imsi1=cv2.filter2D(imsi,-1,LowpassFilterMask)

cv2.imshow("2",imsi1)
