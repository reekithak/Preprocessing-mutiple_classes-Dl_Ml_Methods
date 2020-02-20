#!/usr/bin/env python
# coding: utf-8

# In[255]:


#from sklearn.utlis import shuffle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import PIL
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import warnings
warnings.simplefilter('ignore')
from skimage.io import imread,imshow
from skimage.transform import resize
from skimage.color import rgb2gray
get_ipython().run_line_magic('matplotlib', 'inline')

datagen=ImageDataGenerator(
rotation_range=90
    ,#idth_shift_range=[-200,200],
    #eight_shift_range=0.5,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',brightness_range=[0.2,1.0],zoom_range=[0.5,1.0],
    samplewise_center=True,
    featurewise_center =False,
    #amplewise_std_normalization=
)




lim_cav=17
cav_img=[None]*lim_cav
cav_emp=[None]*lim_cav
cav=os.listdir(r"D:\data_new\cavities")
###################################################
j=0
for i  in cav:
    
    if(j<lim_cav):
        cav_img[j]=imread(r"D:/data_new/cavities/" +i) #reading
        cav_emp[j]=cav_img[j].reshape((1,)+cav_img[j].shape) #reshaping
        j+=1
    else:
        break

        
###################################################

j=0
k=0
for i in cav:
    if(j<lim_cav):
        for batch in datagen.flow(cav_emp[j],batch_size=1,save_to_dir=r'D:\data_new\cavities\augmented',save_prefix='cavities',save_format='jpeg'):
            k+=1
            if(k>100):
                break
        
        j+=1
        k=0
    else:
        break
####################################################
####################################################

lim_disc=20
disc_img=[None]*lim_disc
disc_emp=[None]*lim_disc
disc=os.listdir(r"D:\data_new\discoloration")
###################################################
j=0
for i  in disc:
    
    if(j<lim_disc):
        disc_img[j]=imread(r"D:/data_new/discoloration/" +i) #reading
        disc_emp[j]=disc_img[j].reshape((1,)+disc_img[j].shape) #reshaping
        j+=1
    else:
        break

        
###################################################

j=0
k=0
for i in disc:
    if(j<lim_disc):
        for batch in datagen.flow(disc_emp[j],batch_size=1,save_to_dir=r'D:\data_new\discoloration\augmented',save_prefix='discoloration',save_format='jpeg'):
            k+=1
            if(k>100):
                break
        
        j+=1
        k=0
    else:
        break
####################################################
####################################################


lim_gin=18
gin_img=[None]*lim_gin
gin_emp=[None]*lim_gin
gin=os.listdir(r"D:\data_new\gingivitis")
###################################################
j=0
for i  in gin:
    
    if(j<lim_gin):
        gin_img[j]=imread(r"D:/data_new/gingivitis/" +i) #reading
        gin_emp[j]=gin_img[j].reshape((1,)+gin_img[j].shape) #reshaping
        j+=1
    else:
        break

        
###################################################

j=0
k=0
for i in gin:
    if(j<lim_gin):
        for batch in datagen.flow(gin_emp[j],batch_size=1,save_to_dir=r'D:\data_new\gingivitis\augmented',save_prefix='gingivitis',save_format='jpeg'):
            k+=1
            if(k>100):
                break
        
        j+=1
        k=0
    else:
        break
####################################################
####################################################


lim_gum=19
gum_img=[None]*lim_gum
gum_emp=[None]*lim_gum
gum=os.listdir(r"D:\data_new\gumbleeding")
###################################################
j=0
for i  in gum:
    
    if(j<lim_gum):
        gum_img[j]=imread(r"D:/data_new/gumbleeding/" +i) #reading
        gum_emp[j]=gum_img[j].reshape((1,)+gum_img[j].shape) #reshaping
        j+=1
    else:
        break

        
###################################################

j=0
k=0
for i in gum:
    if(j<lim_gum):
        for batch in datagen.flow(gum_emp[j],batch_size=1,save_to_dir=r'D:\data_new\gumbleeding\augmented',save_prefix='gumbleeding',save_format='jpeg'):
            k+=1
            if(k>100):
                break
        
        j+=1
        k=0
    else:
        break
####################################################
####################################################


    
    



lim_leu=8
leu_img=[None]*lim_leu
leu_emp=[None]*lim_leu
leu=os.listdir(r"D:\data_new\leukamia")
###################################################
j=0
for i  in leu:
    
    if(j<lim_leu):
        leu_img[j]=imread(r"D:/data_new/leukamia/" +i) #reading
        leu_emp[j]=leu_img[j].reshape((1,)+leu_img[j].shape) #reshaping
        j+=1
    else:
        break

        
###################################################

j=0
k=0
for i in leu:
    if(j<lim_leu):
        for batch in datagen.flow(leu_emp[j],batch_size=1,save_to_dir=r'D:\data_new\leukamia\augmented',save_prefix='leukamia',save_format='jpeg'):
            k+=1
            if(k>100):
                break
        
        j+=1
        k=0
    else:
        break
####################################################
####################################################


    
    



lim_perio=19
perio_img=[None]*lim_perio
perio_emp=[None]*lim_perio
perio=os.listdir(r"D:\data_new\leukamia")
###################################################
j=0
for i  in perio:
    
    if(j<lim_perio):
        perio_img[j]=imread(r"D:/data_new/leukamia/" +i) #reading
        perio_emp[j]=perio_img[j].reshape((1,)+perio_img[j].shape) #reshaping
        j+=1
    else:
        break

        
###################################################

j=0
k=0
for i in perio:
    if(j<lim_perio):
        for batch in datagen.flow(perio_emp[j],batch_size=1,save_to_dir=r'D:\data_new\leukamia\augmented',save_prefix='periodnitis',save_format='jpeg'):
            k+=1
            if(k>100):
                break
        
        j+=1
        k=0
    else:
        break
####################################################
####################################################


    
    


