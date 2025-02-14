import cv2 as cv
import numpy as np

def makeGray(img):
    # Converting the BGR image to gray scale
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    return gray

def rescale(frame,scale):
    # Rescaling the image
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    
    dim = (width,height)
    
    new_img = cv.resize(frame,dim,interpolation=cv.INTER_AREA)
    return new_img

def distanceTansform(img):
    # Finding the distance transform of the image
    dist = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            dist[i,j] = min(i,j,img.shape[0]-i,img.shape[1]-j)
    
    # Normalizing the distance transform
    cv.normalize(dist,dist,0,1.0,cv.NORM_MINMAX)
            
    return dist

def blendImages(img1,img2,dist1,dist2,p_w,p_h):
    # Blending the images using the distance transform of the images
    result = np.zeros_like(img1)
    for i in range(p_h):
        for j in range(p_w):
            zero = np.zeros((3))
            w1 = dist1[i,j]
            w2 = dist2[i,j]
            if np.all((w1+w2)!=0):
                result[i,j] = (w1*img1[i,j] + w2*img2[i,j])/(w1+w2)
            else:
                result[i,j] = zero
    return result
    

def detectKeyPoints(img1,img2):
    # Detecting the key points using SIFT
    sift = cv.SIFT_create();
    kp1, desc1 = sift.detectAndCompute(img1,None)
    kp2, desc2 = sift.detectAndCompute(img2,None)
    
    # Matching the key points using Brute Force Matcher
    bf = cv.BFMatcher(cv.NORM_L2,crossCheck=True)
    matches = bf.match(desc1,desc2)
    
    # Sorting the matches based on the distance
    matches = sorted(matches,key =lambda x:x.distance )
    matched_img = cv.drawMatches(img1,kp1,img2,kp2,matches[:75],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    cv.imshow("Feature Matching", matched_img)
    
    return kp1,kp2,matches

def stitchImages(kp1,kp2,matches,img1,img2,gray_img1,gray_img2):
    # Stitching the images
    # Atleast 4 good matches are required to stitch the images
    if len(matches)>4:
        # Finding the homography matrix
        srcPts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        dstPts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
        
        H,_ = cv.findHomography(dstPts,srcPts,cv.RANSAC,5.0)
        
        h1,w1 = img1.shape[:2]
        h2,w2 = img2.shape[:2]
        
        p_w = w1+w2
        p_h = max(h1,h2)
        
        # Warpping the second image to the first image using the homography matrix
        warpped_img2 = cv.warpPerspective(img2, H, (p_w, p_h))
        result = np.zeros_like(warpped_img2)
        result[0:h1, 0:w1] = img1
        
        # Finding the distance transform of the images
        dist_img1 = distanceTansform(gray_img1)
        dist_img2 = distanceTansform(gray_img2)
        
        # Warpping the distance transform of the second image to the first image using the homography matrix
        dist_img2 = cv.warpPerspective(dist_img2, H, (p_w, p_h))
        
        
        # Padding the distance transform of the first image with zeros 
        zero = np.zeros((p_h,p_w))
        zero[0:h1, 0:w1] = dist_img1
        dist_img1 = zero
        
        # Converting the distance transform to 3 channels
        dist_img1 = np.stack((dist_img1,dist_img1,dist_img1),axis=-1)
        dist_img2 = np.stack((dist_img2,dist_img2,dist_img2),axis=-1)
        
        cv.imshow("Distance Transform 1",dist_img1)
        cv.imshow("Distance Transform 2",dist_img2)
    
        # Blending the images
        result = blendImages(result,warpped_img2,dist_img1,dist_img2,p_w,p_h)
        

        return result
    else:
        print('Good maches are less than 4.')
        return None;


def main():
    # To read the 2 images from the images folder
    img1 = cv.imread('images/img1.jpg',cv.IMREAD_COLOR)
    img2 = cv.imread('images/img2.jpg',cv.IMREAD_COLOR)

    # Rescaling the images
    img1 = rescale(img1,0.15)
    img2 = rescale(img2,0.15)

    # Converting the images to gray scale
    gray_img1 = makeGray(img1)
    gray_img2 = makeGray(img2)

    kp1,kp2,matches = detectKeyPoints(gray_img1,gray_img2)

    result = stitchImages(kp1,kp2,matches,img1,img2,gray_img1,gray_img2)


    cv.imshow("Panaroma",result)

    cv.imwrite("Panaroma.jpg", result)  # Save to a JPG file

    cv.waitKey(0)
    
    
if __name__ == "__main__":
    main()