Project Description
Unwanted object removal is a common task in many computer vision and image processing systems. One of the most popular techniques for object removal is image inpainting, which involves filling in the missing or damaged portions of a picture with information from the surrounding area. In this project, we explore methods for removing objects from photos based on inpainting using two approaches: cv2.INPAINT_TELEA, cv2.INPAINT_NS, and averaging neighboring pixels. These are two commonly used OpenCV picture inpainting methods. We compare the results and discuss the underlying principles of these methods, as well as their benefits and drawbacks for eliminating unwanted objects.

Built With :
Tools and Software:
VSCode editor
Google Colab
Climpchamp VideoEditor
MobileCamera
OBS Studio

Getting Started:
Prerequisites
VSCode editor
Operating System: Windows 7/10
Processor: i3/i5 with minimum 8 GB RAM

Installation:
1.create a conda environment
2.activate the environment
3.install the requirements.txt file inside the environment.
4.Open VSCode Editor and run the two .py files attached in the GitHub repository.

Usage and Steps
After installing the required modules:
To run the inpaint code : run the cv_inpaint.py
1.set the input image location path in the codes from the local space.
2.Execute the code by python file name

To run the avg code : run the cv_average.py
1.set the input image location path in the codes from the local space.
2.Execute the code by python file name

To run the segementation code : run the cv_average.py

1.upload the notebook to colab
2.run the code

To run the average code:
1.select the input ad output image path
2. Run the code python cv_project_loopcode
This will open each image one by one 

While running the code for the average function a new window will appear to select the object you want to create the mask.
Click the start point and draw a rectangle of the blue color to select the region.
A new masked image will be shown with the selected area in complete white and the remaining in black.
The final output image will also be displayed, showing the blended image after segmentation.
Press Escape to close all windows or stop the code execution.


while running the Inpaint Function, A new window will open to select the image.
Draw freehand with the mouse to mark the area you want to remove from the image.
After selecting the area, a new window will appear showing the masked image and the final output image.
The two final output images will be saved to your VSCode workspace folder.
Viewing Results in a Series of Images

Press the 'S' key to navigate to the next image in the series.
Select the required area using the mouse and press 'S' to move to the next image.
After processing all the images, go to the workspace folder. You will find all the required blended output images and a CSV file containing PSNR and blending time.
