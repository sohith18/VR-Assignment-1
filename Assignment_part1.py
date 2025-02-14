import cv2 as cv
import numpy as np

# To rescale the image
def rescaleImg(frame, scale):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    
    dim = (width,height)
    
    return cv.resize(frame,dim,interpolation=cv.INTER_AREA)

def main():
    # To read the image
    img = cv.imread('Coins.jpg')

    # Rescaling the image
    resized_img = rescaleImg(img,0.25)

    # Converting the BGR image to gray scale
    gray_img = cv.cvtColor(resized_img, cv.COLOR_BGR2GRAY)

    # Blurring the image using Gaussian Blur using a kernel of size 3x3
    blurred_img = cv.GaussianBlur(gray_img,(3,3),cv.BORDER_REPLICATE)

    # Applying Canny Edge Detection
    canny = cv.Canny(blurred_img,100,150)
    cv.imshow('Canny Edges',canny)

    # Applying Otsu's Thresholding
    _, threshold = cv.threshold(blurred_img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    cv.imshow('Threshold',threshold)

    # Applying Morphological Closing to fill the holes
    kernel = np.ones((5,5), np.uint8)
    closed = cv.morphologyEx(threshold, cv.MORPH_CLOSE, kernel, iterations=6)

    # Finding the contours
    contours, hierarchy = cv.findContours(closed,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if cv.contourArea(cnt)>15000]

    # Creating a mask for the coins
    mask = np.zeros_like(resized_img)
    cv.drawContours(mask,filtered_contours,-1,(0,255,0),thickness=cv.FILLED)
    cv.imshow("Full Mask",mask)

    # Extracting the coins by making a separate mask for each coin and then applying bitwise_and operation
    for i,cnt in enumerate(filtered_contours):
        mask = np.zeros_like(resized_img)
        cv.drawContours(mask,[cnt],-1,(255,255,255),thickness=cv.FILLED)
        
        single_coin = cv.bitwise_and(resized_img,mask)
        
        # Saving the coins to the folder seg_coins
        cv.imwrite(f'seg_coins/coin_{i+1}.png', single_coin)


    print("Number of Coins: ", len(filtered_contours))
    cv.waitKey(0)
    
if __name__ == "__main__":
    main()

