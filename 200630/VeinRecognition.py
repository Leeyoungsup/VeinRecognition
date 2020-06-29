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
def median_filter(image):
        Mask = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        height = image.shape[0]
        width = image.shape[1]
        imageValue = np.zeros([height + 2, width + 2])
        imageValue1 = np.zeros([height + 2, width + 2])
        x = np.arange(width)
        y = np.arange(height)
        x, y = np.meshgrid(x, y)
        imageValue[y + 1, x] = image[y, x]
        imageValue[y + 1, x + 2] = image[y, x]
        imageValue[0] = imageValue[1]
        imageValue[-1] = imageValue[-2]
        for i in range(1, height + 1):
            for j in range(1, width + 1):
                value = Mask * imageValue[i - 1:i + 2, j - 1:j + 2]
                imageValue1[i, j] = np.sort(value.reshape(9))[4]
        imageValue1 = np.delete(imageValue1, width + 1, axis=1)
        imageValue1 = np.delete(imageValue1, height + 1, axis=0)
        imageValue1 = np.delete(imageValue1, 0, axis=1)
        imageValue1 = np.delete(imageValue1, 0, axis=0)
        return imageValue1
imsi=cv2.imread("./2.png",cv2.IMREAD_GRAYSCALE)
imsi=cv2.resize(imsi,dsize=(400,400),interpolation=cv2.INTER_AREA)
imsi1=median_filter(imsi)
imsi1=imsi1.astype(np.uint8)
#imsi1=cv2.filter2D(imsi,-1,LowpassFilterMask)
#ret,imsi1=cv2.threshold(imsi1,100,255,cv2.THRESH_OTSU)
imsi1=cv2.equalizeHist(imsi1)
ret,imsi1=cv2.threshold(imsi1,imsi1.mean(),255,cv2.THRESH_TRUNC)
ret,imsi2=cv2.threshold(imsi1,imsi1.mean(),255,cv2.THRESH_OTSU)
plt.imshow(imsi2,'gray')
plt.show()
