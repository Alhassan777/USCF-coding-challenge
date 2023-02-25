#!/usr/bin/env python
# coding: utf-8

# In[40]:


import numpy as np
from numpy import load #numpy library is used to load the data from the dataset
from matplotlib import pyplot as plt #matplotlib library will be used to graph the pixels to make the final pictures
#loading the dataset from the file downloaded on the hard disk by copying the file path and put it in the function "load"
data = load('/Users/alhassanahmed/Downloads/chestmnist.npz') 
lst=data.files 
print (lst)
# set a varaible that have the names of the files inside the dataset and print it


# In[37]:


first_image = data["test_images"][5] #set the first image to be image number 6 in the test_images file
first_image = np.array(first_image, dtype='float')# convert the lists inside the dataset into array of type float
# this step is essential because plt.imshow() function works on number arrays only
first_image = first_image.reshape((28, 28)) #reshape the image size to be 28 x 28 pixels
plt.imshow(first_image, cmap='gray')#print the image from the number array in the "gray png form"


# In[36]:


# in gray scale images, the pixels values lies in the range of 0-255, which represnets the color levels of the gray color
# 255 represents white and 0 represents black
# by subtracting each pixel from 255, we are making a contrast filter
# for example if a pixel was white, subtracting it from 255 will make a black pixel (255-255 = 0)
# this information was retrived from this source 
# https://www.analyticsvidhya.com/blog/2021/03/grayscale-and-rgb-format-for-storing-images/#:~:text=Now%20let's%20quickly%20summarize%20the,black%20and%20255%20represents%20white.
gray_image= 255-first_image 
gray_image = gray_image_1.reshape((28, 28))
plt.imshow(gray_image, cmap='gray')


# In[28]:


#The original image have two image characteristics called "pixelation" and "edge presence"
# image pixealtion occurs when the photo is enlarged too much so that the pixels appear as large clear squares
# image edges is locations in the image where a significant local change in the image intensity occur
# Thus, we used the Gaussian Filter to blur the image and smooth its edges to make the image more smooth and clear
from scipy.ndimage import gaussian_filter
# Applying the Gaussian filter to the image
filtered_image = gaussian_filter(first_image, sigma=1.5, mode='reflect', cval=0.0, truncate=4.0)
# Display the filtered image
filtered_image = filtered_image.reshape((28, 28))
plt.imshow(filtered_image, cmap='gray')#print the image from the number array in the "gray png form"

# these information where restored from these resources
#https://www.techopedia.com/definition/15902/pixelation#:~:text=Pixelation%20mostly%20occurs%20when%20resizing,blurry%20sections%20in%20the%20image.
#https://cse.usf.edu/~r1k/MachineVisionBook/MachineVision.files/MachineVision_Chapter5.pdf
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html


# In[35]:


second_image = data["test_images"][0] #set the second image to be image number 1 in the test_images file
second_image = np.array(second_image, dtype='float')
second_image = second_image.reshape((28, 28))
plt.imshow(second_image, cmap='gray')
plt.show()


# In[34]:


gray_image_2= 255-second_image
gray_image_2 = gray_image_2.reshape((28, 28))
plt.imshow(gray_image_2, cmap='gray')


# In[31]:


filtered_image_2= gaussian_filter(second_image, sigma=1.5, mode='reflect', cval=0.0, truncate=4.0)
filtered_image_2 = filtered_image_2.reshape((28, 28))
plt.imshow(filtered_image_2, cmap='gray')#print the image from the number array in the "gray png form"


# In[33]:


third_image = data["train_images"][3]#set the third image to be image number 4 in the test_images file
third_image = np.array(third_image, dtype='float')
third_image = third_image.reshape((28, 28))
plt.imshow(third_image, cmap='gray')
plt.show()


# In[32]:


gray_image_3= 255-third_image
gray_image_3 = gray_image_3.reshape((28, 28))
plt.imshow(gray_image_3, cmap='gray')


# In[39]:


filtered_image_3= gaussian_filter(third_image, sigma=1.5, mode='reflect', cval=0.0, truncate=4.0)
filtered_image_3 = filtered_image_3.reshape((28, 28))
plt.imshow(filtered_image_3, cmap='gray')#print the image from the number array in the "gray png form"

