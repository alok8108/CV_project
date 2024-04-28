


import cv2

import numpy as np
import pandas as pd

from skimage import measure

# from PIL import Image
# import the modules
import os
from os import listdir
global df
metrics_file_create = open("Evaluation_metrics_t.csv", "w")
metrics_file_create.write("Filename,MSE,SSIM,PSNR,Inpaint Time\n")
metrics_file_create.close()
df = pd.DataFrame(columns=['Filename','MSE','PSNR','Inpaint Time'])
def append_df(images,mse_val,score,psnr,inpaint_time):
    df = df.append({'Filename':images,'MSE':mse_val,'SSIM':score,'PSNR':psnr,'Inpaint Time': inpaint_time}, ignore_index=True)
    return df
def mse_method(image1, image2):
    #convert the images to grayscale
    img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    h, w = img1.shape
    diff = cv2.subtract(img1, img2)
    mse_val = (diff**2).mean()
    # mse_val = err/(float(h*w))
    return mse_val
def ssim_method(image1, image2):
    #convert the images to grayscale
    print("INSIDE SSIM")
    img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    score=measure.compare_ssim(image1,image2, multichannel = True)
    print("score",score)
    # mse_val = err/(float(h*w))
    return score
# C:\Users\hp\OneDrive\Desktop\LAZYCODER\apartment-lounge-3147892__480.jpg
# get the path/directory

folder_dir = "C:\Users\AVANISH SHUKLA\Downloads\cv project codes\cv project\Images\INPUT IMAGE"
destination_folder_dir = "C:\Users\AVANISH SHUKLA\Downloads\cv project codes\cv project\Images\OUTPUT IMAGE"

global mask,start_point,end_point,dst,metrics_file
# df = pd.DataFrame(columns=['Filename','MSE','SSIM','PSNR','Inpaint Time'])
for images in os.listdir(folder_dir):
    print(images)
    file = folder_dir+images
    print(file)
    original_img1 = cv2.imread(file)
    original_img=cv2.add(original_img1,4)
    mask = np.zeros(original_img.shape[:2], dtype=np.uint8)
    metrics_file = open("Evaluation_metrics_t.csv", "a")
    def mouse_callback(event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            # Draw a circle on the mask

            cv2.circle(mask, (x, y), 10, (255, 255, 255), -1)
            start_point=(x,y)

        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            # Draw a circle on the mask while the mouse button is pressed and moved
            cv2.circle(mask, (x, y), 10, (255, 255, 255), -1)
            cv2.circle(original_img, (x, y), 10, (0, 0, 255), -1)
            end_point=(x,y)

        elif event == cv2.EVENT_LBUTTONUP:
            # Use the selected part of the image as a paint brush to mask the area
            brush = cv2.bitwise_and(original_img, original_img, mask=mask)
            cv2.bitwise_not(mask, mask)
            bg = np.ones_like(original_img, np.uint8) * 255
            cv2.bitwise_not(bg, bg)
            masked_bg = cv2.bitwise_and(bg, bg, mask=mask)
            cut_img = cv2.add(brush, masked_bg)
            original_img[:] = cut_img[:]
            mask.fill(0)

            # Show the modified image
            # cv2.imshow("Modified Image", original_img)
            # original_img[start_point[1]:end_point[1],start_point[0]:end_point[0]]=255
                    #Convert an image from RGB to grayscale mode
            gray_image = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)

            # masked= mask
            # BW= Image.convert('1')
            # print(start_point,end_point)

            #Convert a grayscale image to black and white using binary thresholding
            # (thresh, BnW_image) = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV)
            (thresh, BnWmasked_image) = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)
            # cv2.imshow("Modified Image", BnWmasked_image)
            # masked=original_img - BnWmaske  d_image
            # cv2.imshow("MaSKKK",masked)
            # (thresh, BnWmasked_image1) = cv2.threshold(masked, 0, 255, cv2.THRESH_BINARY)
            # cv2.imshow("MaSKKK2",BnWmasked_image1)
            # cv2.imshow('Masked Image in pure black and white ', BnWmasked_image)
            #USING inpaint fn to just blend the empty portion with its surroundings
            e1 = cv2.getTickCount()
            dst = cv2.inpaint(original_img1,BnWmasked_image,3,cv2.INPAINT_TELEA)
            cv2.destroyAllWindows()
            e2 = cv2.getTickCount()
            # cv2.imshow('THE IMAGE BLENDED WITH ITS BACKGROUND WITHOUT THE OBJECT',dst)s
            cv2.imshow('THE IMAGE BLENDED WITH ITS BACKGROUND WITHOUT THE OBJECT',dst)

            # cv2.imwrite("Masked Image",BnWmasked_image)
            # cv2.imwrite("Masked Image",masked_image)
            cv2.imwrite(images, dst)
            inpaint_time = (e2 - e1)/ cv2.getTickFrequency()
            psnr = cv2.PSNR(original_img, dst, 255)
            print("printing values")
            print(images,ssim_method(original_img,dst),psnr,inpaint_time)
            data=images+","+str(mse_method(original_img,dst))+","+str(ssim_method(original_img,dst))+str(psnr)+","+str(inpaint_time)+"\n"
            print(data)
            metrics_file.write(data)
            # df = df.append({'Filename':images,'MSE':mse_method(original_img,dst),'PSNR':psnr,'Inpaint Time': inpaint_time}, ignore_index=True)
            # print("printing df")
            # print(writable_df)
            # writable_df.to_csv('Evaluation_metrics.csv')
        # cv2.destroyAllWindows()


    cv2.namedWindow("Drag like a paint brush the mouse cursor to select the portion of Image")
    cv2.setMouseCallback("Drag like a paint brush the mouse cursor to select the portion of Image", mouse_callback)
    # cv2.imshow("Drag like a paint brush the mouse cursor to select the portion of Image", original_img)
    while True:
        key = cv2.waitKey(1)

        # If the 's' key is pressed, exit the loop
        if key == ord('s'):
            break

        # Show the mask in real time
        mask_display = cv2.bitwise_and(original_img, original_img, mask=mask)
        cv2.imshow("Mask", mask_display)
        cv2.imshow("Drag like a paint brush the mouse cursor to select the portion of Image", original_img)
        # cv2.imshow('Masked Image in pure black and white ', BnWmasked_image)
        # cv2.imshow('THE IMAGE BLENDED WITH ITS BACKGROUND WITHOUT THE OBJECT',dst)
        # cv2.imshow("")

        # Show the original image
        # cv2.imshow("Image", original_img)







