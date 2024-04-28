

# Importing required libraries
from skimage.segmentation import slic
from skimage.data import astronaut
from skimage.color import label2rgb
import matplotlib.pyplot as plt

# Setting the plot size as 15, 15
plt.figure(figsize=(15,15))

# Sample Image of scikit-image package
astronaut = astronaut()

# Applying Simple Linear Iterative
# Clustering on the image
# - 50 segments & compactness = 10
astronaut_segments = slic(astronaut,
						n_segments=50,
						compactness=10)
plt.subplot(1,2,1)

# Plotting the original image
plt.imshow(astronaut)
plt.subplot(1,2,2)

# Converts a label image into
# an RGB color image for visualizing
# the labeled regions.
plt.imshow(label2rgb(astronaut_segments,
					astronaut,
					kind = 'avg'))



import matplotlib.pyplot as plt
import matplotlib.image as npimage
import numpy as np
from skimage import color
import cv2 as cv
# from skimage.filters import filters
from skimage.transform import hough_circle
from skimage.feature import peak_local_max
from skimage import feature
from skimage import morphology
from skimage.draw import  circle_perimeter
from skimage import img_as_float, img_as_ubyte
from skimage import segmentation as seg
# from skimage.morphology import watershed
from scipy.ndimage import convolve
import glob
from skimage.data import astronaut
import cv2 as cv
from google.colab.patches import cv2_imshow


plt.rcParams['image.cmap'] ='gray'
plt.rcParams['image.interpolation'] = 'none'
plt.rcParams['figure.figsize'] = 10,10
img=astronaut()
np.unique(img[0])

plt.rcParams['image.cmap'] ='gray'
plt.rcParams['image.interpolation'] = 'none'
plt.rcParams['figure.figsize'] = 10,10
# img=cv.imread("/content/1_0ttFuc9a8RMCQxCbrNeMNg.png",0)
img=astronaut()
np.unique(img)

# imgpaths = glob.glob()

def mean_color(image, labels):
  out = np.zeros_like(image)
  for label in np.unique(labels):
    indices = np.nonzero(labels == label)
    out[indices] = np.mean(image[indices], axis=0)
  return out
def find_mean_indices(labels):
  # mean_indices = [2,len(np.unique(labels))]
  mean_indices=[]
  for label in np.unique(labels):
    indices = np.nonzero(labels == label)
    mean_x_index=int(np.sum(indices[0])/indices[0].shape[0])
    mean_y_index=int(np.sum(indices[1])/indices[1].shape[0])
    mean_indices.append([mean_x_index,mean_y_index])
  return mean_indices

# def find_mean_indices(labels):
#   # mean_indices = [2,len(np.unique(labels))]
#   mean_indices=[]
#   for label in np.unique(labels):
#     indices = np.nonzero(labels == label)
#     mean_x_index=int(np.sum(indices[0])/512*512)
#     print(indices[0])
#     print(indices[0].shape[0])
#     mean_y_index=int(np.sum(indices[1])/512*512)
#     mean_indices.append([mean_x_index,mean_y_index])
#   return mean_indices
# Find longest x sum and longest y sum and then take the mid value. Need to find longest x sum and longest y sum for each label
# for each label, for each y, find longest x sum and sae for y sum.

def plot_slic_segmentation(img, ns, c, s):
  labels = seg.slic(img, n_segments=ns, compactness=c, sigma=s, enforce_connectivity=True)
  labels+=1
  return mean_color(img, labels),labels
def mark_labels(label_img,mean_indices):
  label_count=1
  print(mean_indices)
  for position in mean_indices:
    label_text = str(label_count)
    label_img = cv.putText(label_img,label_text,position,cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
    label_count+=1
  plt.imshow(label_img)




ns=15
compact=50
sigma=0.1

rgbimage=img_as_float(color.gray2rgb(img))
plt.figure()
plt.subplot(1,2,1)
plt.imshow(img)
plt.subplot(1,2,2)
seg_img,labels=plot_slic_segmentation(img,ns,compact,sigma)
mean_indices=find_mean_indices(labels)
mark_labels(seg_img,mean_indices)
plt.imshow(seg_img)

ns=15
compact=50
sigma=0.1

rgbimage=img_as_float(color.gray2rgb(img))
plt.figure()
plt.subplot(1,2,1)
plt.imshow(img)
plt.subplot(1,2,2)
seg_img,labels=plot_slic_segmentation(img,ns,compact,sigma)
mean_indices=find_mean_indices(labels)
mark_labels(seg_img,mean_indices)
plt.imshow(seg_img)
# plt.imshow(plot_slic_segmentation(rgbimage,ns,compact,sigma))
plt.imshow(labels)

labels = seg.slic(img, n_segments=25, sigma=0.2, enforce_connectivity=True)
labels

astro_segs = plot_slic_segmentation(rgbimage,25,70,0)
astro_segs.ndim

np.unique(astro_segs[0])











