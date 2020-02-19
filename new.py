#!/usr/bin/env python
#coding: utf-8




import numpy as np
import pandas as pd
import tensorflow as tf
from keras import *
import matplotlib.pyplot as plt
import cv2
import os 
import warnings
warnings.simplefilter('ignore')
from skimage.io import imread,imshow
from skimage.transform import resize
from skimage.color import rgb2gray
get_ipython().run_line_magic('matplotlib', 'inline')





lim_cav=17
lim_disc=20
lim_gin=18
lim_gum=19
lim_leu=8
lim_perio=19


 #6 CLASSES




cav=os.listdir(r"E:\dataset_medical\cavities")
disc=os.listdir(r"E:\dataset_medical\discoloration")
gin=os.listdir(r"E:\dataset_medical\gingivitis")
gum=os.listdir(r"E:\dataset_medical\gumbleeding")
leu=os.listdir(r"E:\dataset_medical\leukamia")
perio=os.listdir(r"E:\dataset_medical\perio")

#empty lists 
cav_img=[None]*lim_cav
disc_img=[None]*lim_disc
gin_img=[None]*lim_gin
gum_img=[None]*lim_gum
leu_img=[None]*lim_leu
perio_img=[None]*lim_perio

#grayscales - empty

cav_gray=[None]*lim_cav
disc_gray=[None]*lim_disc
gin_gray=[None]*lim_gin
gum_gray=[None]*lim_gum
leu_gray=[None]*lim_leu
perio_gray=[None]*lim_perio


#importing images 





j=0
for i  in cav:
    
    if(j<lim_cav):
        cav_img[j]=imread(r"E:/dataset_medical/cavities/" +i)
        j+=1
    else:
        break

j=0
for i  in disc:
    
    if(j<lim_disc):
        disc_img[j]=imread(r"E:/dataset_medical/discoloration/" +i)
        j+=1
    else:
        break

j=0
for i  in gin:
    
    if(j<lim_gin):
        gin_img[j]=imread(r"E:/dataset_medical/gingivitis/" +i)
        j+=1
    else:
        break

j=0
for i  in gum:
    
    if(j<lim_gum):
        gum_img[j]=imread(r"E:/dataset_medical/gumbleeding/" +i)
        j+=1
    else:
        break

j=0
for i  in leu:
    
    if(j<lim_leu):
        leu_img[j]=imread(r"E:/dataset_medical/leukamia/" +i)
        j+=1
    else:
        break

j=0
for i  in perio:
    
    if(j<lim_perio):
        perio_img[j]=imread(r"E:/dataset_medical/perio/" +i)
        j+=1
    else:
        break



j=0
for i  in cav:
    
    if(j<lim_cav):
        cav_gray[j]=rgb2gray(cav_img[j])
        j+=1
    else:
        break

j=0
for i  in disc:
    
    if(j<lim_disc):
        disc_gray[j]=rgb2gray(disc_img[j])
        j+=1
    else:
        break

j=0
for i  in gin:
    
    if(j<lim_gin):
        gin_gray[j]=rgb2gray(gin_img[j])
        j+=1
    else:
        break

j=0
for i  in gum:
    
    if(j<lim_gum):
        gum_gray[j]=rgb2gray(gum_img[j])
        j+=1
    else:
        break

j=0
for i  in leu:
    
    if(j<lim_leu):
        leu_gray[j]=rgb2gray(leu_img[j])
        j+=1
    else:
        break

j=0
for i  in perio:
    
    if(j<lim_perio):
        perio_gray[j]=rgb2gray(perio_img[j])
        j+=1
    else:
        break





for j in range(lim_cav):
    cav_gray[j]=resize(cav_gray[j],(252,252))
for j in range(lim_disc):
    disc_gray[j]=resize(disc_gray[j],(252,252))
for j in range(lim_gin):
    gin_gray[j]=resize(gin_gray[j],(252,252))
for j in range(lim_gum):
    gum_gray[j]=resize(gum_gray[j],(252,252))
for j in range(lim_leu):
    leu_gray[j]=resize(leu_gray[j],(252,252))
for j in range(lim_perio):
    perio_gray[j]=resize(perio_gray[j],(252,252))


#Flattening the grayscales





len_of_cav=len(cav_gray)
im_size_cav=cav_gray[1].shape
flat_size_cav=im_size_cav[0] * im_size_cav[1]

for i in range(len_of_cav):
    cav_gray[i]=np.ndarray.flatten(cav_gray[i]).reshape(flat_size_cav,1)
cav_gray=np.dstack(cav_gray)


len_of_disc=len(disc_gray)
im_size_disc=disc_gray[1].shape
flat_size_disc=im_size_disc[0] * im_size_disc[1]

for i in range(len_of_disc):
    disc_gray[i]=np.ndarray.flatten(disc_gray[i]).reshape(flat_size_disc,1)
disc_gray=np.dstack(disc_gray)


len_of_gin=len(gin_gray)
im_size_gin=gin_gray[1].shape
flat_size_gin=im_size_gin[0] * im_size_gin[1]

for i in range(len_of_gin):
    gin_gray[i]=np.ndarray.flatten(gin_gray[i]).reshape(flat_size_gin,1)
gin_gray=np.dstack(gin_gray)


len_of_gum=len(gum_gray)
im_size_gum=gum_gray[1].shape
flat_size_gum=im_size_gum[0] * im_size_gum[1]

for i in range(len_of_gum):
    gum_gray[i]=np.ndarray.flatten(gum_gray[i]).reshape(flat_size_gum,1)
gum_gray=np.dstack(gum_gray)


len_of_leu=len(leu_gray)
im_size_leu=leu_gray[1].shape
flat_size_leu=im_size_leu[0] * im_size_leu[1]

for i in range(len_of_leu):
    leu_gray[i]=np.ndarray.flatten(leu_gray[i]).reshape(flat_size_leu,1)
leu_gray=np.dstack(leu_gray)


len_of_perio=len(perio_gray)
im_size_perio=perio_gray[1].shape
flat_size_perio=im_size_perio[0] * im_size_perio[1]

for i in range(len_of_perio):
    perio_gray[i]=np.ndarray.flatten(perio_gray[i]).reshape(flat_size_perio,1)
perio_gray=np.dstack(perio_gray)



#Roll Axis and Reshape




cav_gray=np.rollaxis(cav_gray,axis=2,start=0)

cav_gray=cav_gray.reshape(len_of_cav,flat_size_cav)



disc_gray=np.rollaxis(disc_gray,axis=2,start=0)

disc_gray=disc_gray.reshape(len_of_disc,flat_size_disc)



gin_gray=np.rollaxis(gin_gray,axis=2,start=0)

gin_gray=gin_gray.reshape(len_of_gin,flat_size_gin)



gum_gray=np.rollaxis(gum_gray,axis=2,start=0)

gum_gray=gum_gray.reshape(len_of_gum,flat_size_gum)



leu_gray=np.rollaxis(leu_gray,axis=2,start=0)

leu_gray=leu_gray.reshape(len_of_leu,flat_size_leu)




perio_gray=np.rollaxis(perio_gray,axis=2,start=0)

perio_gray=perio_gray.reshape(len_of_perio,flat_size_perio)


#Making dataframes




cav_data=pd.DataFrame(cav_gray) ; cav_data['Label']='Cavity'
disc_data=pd.DataFrame(disc_gray) ; disc_data['Label']='Discolouration'
gin_data=pd.DataFrame(gin_gray) ; gin_data['Label']='Gingivitis'
gum_data=pd.DataFrame(gum_gray) ; gum_data['Label']='Gum bleeding'
leu_data=pd.DataFrame(leu_gray) ; leu_data['Label']='leukamia'
perio_data=pd.DataFrame(perio_gray) ; perio_data['Label']='periodontitis'


#MAking / concatenating the data frames




ac_1=pd.concat([cav_data,disc_data])
ac_2=pd.concat([gin_data,gum_data])
ac_3=pd.concat([leu_data,perio_data])
ac_m=pd.concat([ac_1,ac_2])
ac=pd.concat([ac_m,ac_3])



#SHuffling





from sklearn.utils import shuffle





T_diseases=shuffle(ac).reset_index()





T_diseases=T_diseases.drop(['index'],axis=1)





T_diseases


#Make a csv file





T_diseases.to_csv(r"E:/dataset_medical/diseases.csv")








#DUMB TESTING and SPLITTING




from sklearn import *
from sklearn.model_selection import train_test_split





x=T_diseases.values[:,:-1]
y=T_diseases.values[:,-1]





x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=True)


#using svm_ 




clf = svm.SVC(gamma=0.001)





clf.fit(x_train,y_train)





y_pred = clf.predict(x_test)





print("classifier report  %s:\n%s\n"
      % (classifier, metrics.classification_report(y_test, y_pred)))













