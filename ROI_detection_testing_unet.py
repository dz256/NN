#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
#matplotlib notebook
#%pdb

import tensorflow as tf
import sys
import scipy.io as sio
import scipy.stats as stats
import scipy.signal as signal
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
#import tensorflow as tf
import time
import math
import ipywidgets
import PIL
#import itertools
import os
from numpy.lib.format import open_memmap
from tqdm import tqdm
from tqdm import tqdm_notebook as tqdm_notebook
#from numba import jit

#import cv2

#from ipywidgets import FloatProgress
from IPython.display import display


# In[ ]:


from keras.models import Sequential, load_model
from keras.layers import Input
from keras.layers import Dense, Dropout, Activation, Flatten, Concatenate
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.core import Lambda
import keras.optimizers as optimizers
from keras import regularizers
from keras import backend as K
from keras.models import Model


# In[ ]:


#from build_dataset import roi_unet_test as dataset
from build_dataset import test_data_generator
from build_model import load_model_weights, load_saved_model


# In[ ]:


os.environ['CUDA_VISIBLE_DEVICES'] = '2'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.log_device_placement = True
sess = tf.Session(config=config)
K.set_session(sess)


# In[ ]:


weights_filename = 'ROI_detection_unet_20190416-151650_best_weight.weight'
model = load_saved_model(weights_filename)


# model_folder = 'ROI_detection_unet_20170808-185834'
# model_filename = 'ROI_detection_unet_20170808-185834_20170809_131239_99.keras'
# model = load_model('X:\Hua-an\Data\Deep_Learning_Model\%s\%s.keras' % (model_folder,model_folder))

# In[ ]:


print(model.summary())


# In[ ]:


#test_dataset_folder = ["C:\\Users\\Hua-an\\Documents\\CloudStation\\Lab_Project\\Ultrasound\\Analysis\\max-min_image\\control_hippocampus\\New folder"]


# In[ ]:


test_dataset_folder = ["E:\\temp\\Munib"]


# In[ ]:


params = {'model': model,
          'stride_ratio': 0.5,
          'flip': False,
          'rotate': False,
          'std_normalization': True,
         }
test_generator = test_data_generator(test_dataset_folder, **params)


# In[ ]:


test_generator.predict()


# In[ ]:


test_generator.create_roi(roi_threshold=0.5, roi_size_limit=[50, 500], erosion_iter=1, watershed=True)


# In[ ]:


test_generator.save_predict_roi()


# In[ ]:


test_generator.show_predict_roi(file_idx=0,area_idx=None, colors='red',linewidths=0.5)


# In[ ]:


test_generator.show_predict_roi(file_idx=1,area_idx=None, colors='red',linewidths=0.5)


# In[ ]:


test_generator.show_predict_roi(file_idx=1,area_idx=[[600,800],[400,600]], colors='red',linewidths=0.5)


# In[ ]:


test_generator.show_predict_roi(file_idx=1,area_idx=[[600,800],[900,1100]], colors='red',linewidths=0.5)


# In[ ]:


test_generator.show_predict_roi(file_idx=2,area_idx=[[600,800],[200,400]], colors='red',linewidths=0.5)


# In[ ]:


test_generator.show_predict_roi(file_idx=3,area_idx=None, colors='red',linewidths=0.5)


# In[ ]:


test_generator.show_predict_roi(file_idx=4,area_idx=None, colors='red',linewidths=0.5)


# In[ ]:


test_generator.show_predict_roi(file_idx=5,area_idx=None, colors='red',linewidths=0.5)

