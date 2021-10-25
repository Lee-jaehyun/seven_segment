from PIL.Image import Image
import cv2
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import numpy as np
from numpy.lib.function_base import disp
from source import models
from display_select import select
from seven_segocr import transform_image, predict_model
import torch


new_model = 'resnet20' # resnet 구조 사용 가능, resnet8, resnet14, ... or resnet18, resnet34, resnet50 ...

new_model = models.__dict__[new_model](num_classes=10)

#print(new_model)

new_model.load_state_dict(torch.load('source/model_state_dict.pt'))


# 명암대비 함수
def increase_contrast(image, clipLimit=2, size=(3,3)):
    clahe = cv2.createCLAHE(clipLimit, tileGridSize=size)     # default = 2
    image_contrast = clahe.apply(image)
    return image_contrast


# 숫자 컨투어 정렬함수
# 순서대로 정렬
def sort_contours(digitCnts, image):
    digit = []
    sorted_ctrs = sorted(digitCnts, key=lambda ctr: (cv2.boundingRect(ctr)[1] + cv2.boundingRect(ctr)[0]))   #cv2.boundingRect(ctr)[0]

    for i, ctr in enumerate(sorted_ctrs):
    # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)

    # Getting ROI
        roi = image[y:y+h, x:x+w]
        digit.append(roi)

    # show ROI
        cv2.imshow("roi", roi)
        cv2.waitKey(0)
    return digit



# 이미지 전처리 하는 방법 (디스플레이 영역 검출)
def preprocess(path):
    image = cv2.imread(path)
    h, w = image.shape[:2]
    print(w, h)
 #   image = image[int(h*1/5):int(h - (h*1/6)), :]

    image_resized = imutils.resize(image=image, height=1200, width=1000)    # 가변
    image_resized2 = image_resized.copy()

    cv2.imshow("image", image_resized2)
    cv2.waitKey(0)

    grayscale = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

    grayscale = cv2.GaussianBlur(grayscale, (3, 3), 0)

    contrast_grayscale = increase_contrast(grayscale, clipLimit=7, size=(5,5))

    #th, src_bin = cv2.threshold(contrast_grayscale, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    image_binary = cv2.adaptiveThreshold(contrast_grayscale, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

   # th, image_binary = cv2.threshold(image_binary, 45, 255, cv2.THRESH_BINARY_INV)

   # kernel2 = np.ones((5, 3), np.uint8)
   # src_bin = cv2.dilate(src_bin, kernel2, iterations=3)


   # cv2.imshow("contrast", image_binary)
   # cv2.waitKey(0)

    blurred = cv2.GaussianBlur(contrast_grayscale, (3, 3), 0)
    blurred = cv2.GaussianBlur(blurred, (5, 5), 0)

  #  cv2.imshow("blur_before_canny", blurred)
  #  cv2.waitKey(0)

    canny = cv2.Canny(contrast_grayscale, 40, 160 , 255)

    cv2.imshow("canny", canny)
    cv2.waitKey(0)

    cnts, hierarchy = cv2.findContours(image=canny, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)

    mask = np.zeros(canny.shape, dtype=np.uint8)
    test = image_resized2
    test_images = []
    
    for idx in range(len(cnts)):
        x,y,w,h = cv2.boundingRect(cnts[idx])
        mask[y:y+h, x:x+w] = 0
        cv2.drawContours(mask, cnts, idx, (255, 255, 255), -1)
        r = float(cv2.countNonZero(mask[y:y+h, w:w+x]))
        if r > 0.3 and (100 < w) and (300 < h <800):     # r>0.3
            if (y-5 < 0) or (x-3 < 0):
                continue
            test_image = test[y:y+h, x:x+w]                     #### test[y-5:y+h+15, x-3:x+w+3]
            print("[INFO] : test_image.shape :", test_image.shape, "========", w, h)
            test_images.append(test_image)

            cv2.imshow("test_image", test_image)
            cv2.waitKey(0)

    if len(test_images)==0:
        print("Not to find Display!!")

    test_images = sorted(test_images, key= lambda x: -(x.shape[0]*x.shape[1]))

    return test_images

  #  cv2.imshow("image", canny)
  #  cv2.waitKey(0)


## 숫자 영역 검출
def roi_seg(image):
    if image == "None":
        return None

    image = imutils.resize(image=image, height=1200, width=1200)

    test_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.GaussianBlur(test_gray, (3, 3), 0)
    test_gray = cv2.GaussianBlur(test_gray, (5, 5), 0)

    test_gray = increase_contrast(test_gray, clipLimit=15, size=(3,3))   # 2

    test_gray = cv2.GaussianBlur(test_gray, (3, 3), 0)
    test_gray = cv2.GaussianBlur(test_gray, (5, 5), 0)
   # test_gray = cv2.GaussianBlur(test_gray, (3, 3), 0)


    #thresh = cv2.threshold(test_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    _, thresh = cv2.threshold(test_gray, 77, 255, cv2.THRESH_BINARY_INV)

    cv2.imshow("thresh", thresh)
    cv2.waitKey(0)

   # kernel3 = np.ones((10, 3), np.uint8)
  #  thresh = cv2.dilate(thresh, kernel3, iterations=1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 5))    # (1,5)
   
    kernel2 = np.ones((10,2), np.uint8)                       # (세로, 가로)     # (28, 4)
    thresh2 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    cv2.imshow("thresh2222", thresh2)
    cv2.waitKey(0)

    thresh2 = cv2.dilate(thresh2, kernel2, iterations=2)
    print(thresh2.shape)

    cv2.imshow("thresh2", thresh2)
    cv2.waitKey(0)

    # resize
   # thresh2 = imutils.resize(image=thresh2, height=800, width=1000)      # 600, 800

    cnts = cv2.findContours(thresh2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # thresh     # CHAIN_APPROX_NONE      CHAIN_APPROX_SIMPLE
    cnts = imutils.grab_contours(cnts)

    print("[INFO] : 숫자 검출 :", len(cnts))

    digitCnts = []

    # loop over the digit area candidates
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)

        # if the contour is sufficiently large, it must be a digit
        if (50 <= w <400) and (400<h<700) and ((0.3 <w/h) or (0.17 > w/h)):  # 30 ,50                        80, 12, 200       40     / 300 / 500
            digitCnts.append(c)   #c
      #      print(x)
      #      print(y)
      #      print(x + y)
      #     # x1, y1, w1, h1 = x, y, w, h

            cv2.imshow("digit", thresh2[y:y + h, x:x + w])
            cv2.waitKey(0)

    a = sort_contours(digitCnts=digitCnts, image=thresh)
    print("=============", a[1])
    return a

def predict_number(roi_list):
    number_list = []
    print("갯수 :", len(roi_list))
    for i in range(len(roi_list)):
        #cv2.imshow('img', img)
        #cv2.waitKey(0)
        test_d = transform_image(roi_list[i])   ###   glu.png

        number = predict_model(new_model, test_d)

        number_list.append(int(number))
    return number_list


if __name__ == "__main__":
    #display = preprocess('../../../../Volumes/USB/data/train/체중계/k012.jpg')         #'../../../../Volumes/USB/data/train/체온계/j001.jpg'     ../../../../Volumes/USB/data/train/혈당계/p001.jpg
   # display = preprocess('../data/train/체중계/j032.jpg')
  #  display = preprocess('../cheon_test7.jpeg')
   # display = preprocess('../data/train/혈압계/m019.jpg')
    display = preprocess('../data/train/혈압계/j015.jpg')

    test_img = select(display)
    roi_list = roi_seg(test_img)
    print(list(predict_number(roi_list)))

    print("[INFO] ==> Process Success!")