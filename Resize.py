import matplotlib.pyplot as plt
import numpy as np
import cv2
from IMGShow import IMGShow

class QualityEnhancement :
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    Identity = np.array([[0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0]])
    
    GuassianBlur = (1 / 16) * np.array([[1, 2, 1],
                                        [2, 4, 2],
                                        [1, 2, 1]])
    
    img_sharpen = cv2.filter2D(IMGShow.img_mp1, -1, kernel)

def ResizeImage() :
    img_reize = cv2.resize(IMGShow.img_mp1, (240, 240))
    plt.imshow(img_reize)
    plt.title('Resized Cat Image to 240x240')
    #plt.show()
    return ResizeImage

def SharpenImage() :
    plt.imshow(QualityEnhancement.img_sharpen)
    plt.title('Sharpened Cat Image using Convolution Kernel')
    #plt.show()
    return SharpenImage

def IdentityImage() :
    img_identity = cv2.filter2D(IMGShow.img_mp1, -1, QualityEnhancement.Identity)
    plt.imshow(img_identity)
    plt.title('Identity Filter Applied to Cat Image')
    #plt.show()
    return IdentityImage

def GuassianBlurImage() :
    img_blur = cv2.filter2D(IMGShow.img_mp1, -1, QualityEnhancement.GuassianBlur)
    plt.imshow(img_blur)
    plt.title('Guassian Blur Applied to Cat Image')
    plt.show()
    return GuassianBlurImage

if __name__ == "__main__":
    SharpenImage()
    IdentityImage()
    GuassianBlurImage()
