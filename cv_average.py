

import cv2
import numpy as np
from skimage import measure
# from PIL import Image
basic_img = cv2.imread(r'C:\Users\AVANISH SHUKLA\Downloads\cv project codes\Images\INPUT IMAGE\apartment-lounge-3147892__480.jpg', cv2.IMREAD_COLOR)
img=basic_img
# print(img)
img2=img
# cvimshow("image",img)
# img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
# e1 = cv2.getTickCount()
def mse_method(image1, image2):
    #convert the images to grayscale
    img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    h, w = img1.shape
    print(h,w)
    # diff = cv2.absdiff(img1, img2)
    # print("diff",diff[380])
    # mse_val = (diff**2).mean()
    # print("mse",mse_val)
    # mse_val = err/(float(h*w))
    # return mse_val
# def ssim_method(image1, image2):
    #convert the images to grayscale
    # print("INSIDE SSIM")
    img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # score=measure.compare_ssim(image1,image2, multichannel = True)
    # print("score")
    # mse_val = err/(float(h*w))
    # return score
def calculate_average(pixel,image):
    # Calculate the periphery of the pixel
    x, y = pixel
    periphery = image[x-2:x-1, y-2:y-1]

    # Calculate the average of the periphery
    avg = periphery.mean(axis=0).mean(axis=0)

    return avg
def blend_image(start_point, end_point, k,image):
    # Loop over each pixel

    for x in range(start_point[1], end_point[1]):
        for y in range(start_point[0], end_point[0]):
            # Calculate the average of the previous pixels
            avg = calculate_average((x, y),image)

            # Set the pixel value to the average
            image[x, y] = avg
    return image


def mouse_callback(event, x, y, flags, params):
    global start_point, end_point, drawing, basic_img

    if event == cv2.EVENT_LBUTTONDOWN:
        start_point = (x, y)
        drawing = True

    elif event == cv2.EVENT_LBUTTONUP:
        end_point = (x, y)
        drawing = False

        # Create a mask of the selected part of the image
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.rectangle(mask, start_point, end_point, (255, 255, 255), -1)

        # Display the mask
        masked_image = cv2.bitwise_and(img, img, mask=mask)
        # cv2.imshow('Masked Image ', masked_image)
        #Convert an image from RGB to grayscale mode
        gray_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
        gray_image[start_point[1]:end_point[1],start_point[0]:end_point[0]]=255

        #Convert a grayscale image to black and white using binary thresholding
        (thresh, BnW_image) = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)
        (thresh, BnWmasked_image) = cv2.threshold(BnW_image, 0, 255, cv2.THRESH_BINARY)
        cv2.imshow('Masked Image in pure black and white ', BnWmasked_image)
        #USING inpaint fn to just blend the empty portion with its surroundings
        e1 = cv2.getTickCount()
        blendedimg=blend_image(start_point, end_point, 1,img)
        e2 = cv2.getTickCount()
        inpaint_time = (e2 - e1)/ cv2.getTickFrequency()
        print("inpaint time:",inpaint_time)
        # ssim_method(img2,img)
        # print("MSE error:",mse_method(img2,blendedimg))
        psnr = cv2.PSNR(basic_img, img, 255)
        print("PSNR",psnr)

        # dst = cv2.inpaint(img,BnWmasked_image,3,cv2.INPAINT_TELEA)
        cv2.imshow('THE IMAGE BLENDED WITH ITS BACKGROUND WITHOUT THE OBJECT',blendedimg)
        cv2.imwrite('FINAL OUTPUT OF BlendInvision Image.jpg', img)
        cv2.imwrite("Masked Image",BnWmasked_image)
        cv2.imwrite("Masked Image",masked_image)




        #Display the original image without the selected object
        # cv2.imshow('THE IAMGE WITHOUT THE OBJECT ',(img-masked_image))

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        img_copy = np.copy(img)
        #draws the selected rectangle in the copied image and proceed to draw further to the position of (x,y)
        #draw the rectangle in such a manner that end point coordinate must have greater positional
        #value then start points in x and y direction considering centre to the top left part of the image
        cv2.rectangle(img_copy, start_point, (x, y), (255, 0, 0), 2)
        cv2.imshow('Image', img_copy)



cv2.namedWindow('Drag the mouse cursor to select the portion of Image')
cv2.setMouseCallback('Drag the mouse cursor to select the portion of Image', mouse_callback)
cv2.imshow('Drag the mouse cursor to select the portion of Image', img)
cv2.waitKey(0)
# Create a black image of the same size as the original image
# mask = np.zeros(img.shape[:2], dtype=np.uint8)
# masked_image = cv2.bitwise_and(img_copy, img_copy, mask=mask)
# # Display the masked image
# cv2.imshow('Masked Image', (masked_image))





