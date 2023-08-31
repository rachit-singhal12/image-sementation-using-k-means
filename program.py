import pandas as pd
import cv2
im = cv2.imread("C:\\Users\\lenovo\\Desktop\\completed projects\\image segmentation using k-means\\image.webp")
im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
os = im.shape
#cv2.imshow("image",im)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

all_pixels = im.reshape((493*800,3))
print(all_pixels)

#importin the kmean from sklearn
from sklearn.cluster import KMeans
import numpy as np
color = 4
km = KMeans(n_clusters = color,n_init='auto')
km.fit(all_pixels) 

center = km.cluster_centers_

center = np.array(center,dtype='uint8')

colors=[]
i=1

import matplotlib.pyplot as plt
plt.figure(0,figsize=(4,2))
for each_col in center:
    plt.subplot(1,4,i)
    i+=1
    colors.append(each_col)

    a=  np.zeros((100,100,3),dtype='uint8')
    a[:,:,:] = each_col
    
    plt.imshow(a)

#plt.show()


new_img  = np.zeros((493*800,3),dtype='uint8')

for ix in range(new_img.shape[0]):
    new_img[ix] = colors[km.labels_[ix]]

new_img = new_img.reshape(os)
cv2.imshow("image",new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()