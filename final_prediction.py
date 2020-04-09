#!/usr/bin/env python
# coding: utf-8

# In[91]:


import numpy as np
import cv2
import os
import tqdm

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.models import load_model


# In[103]:


test_path = '/kaggle/input/leaf-classification/dataset/test'
path_res_model = '/kaggle/input/reg-resnet50/reg_model.h5'
path_custom_model = '/kaggle/input/custom-model-file/leaf_reco.h5'


# In[115]:



batch_size_res = 64
batch_size_custom = 8


test_datagen = ImageDataGenerator(rescale=1./255, )

test_datagen_tta = ImageDataGenerator(rescale=1./255,  horizontal_flip=True, 
                                      vertical_flip=True,  
                                      brightness_range=[0.5,1.0])


validation_generator_custom = test_datagen.flow_from_directory(test_path,
                                                       target_size = (64,64),
                                                       shuffle = False,
                                                        batch_size = batch_size_custom,
                                                       class_mode = 'categorical')

validation_generator_res = test_datagen.flow_from_directory(test_path,
                                                       target_size = (224,224),
                                                       shuffle = False,
                                                        batch_size = batch_size_res,
                                                       class_mode = 'categorical')

validation_generator_res_tta = test_datagen_tta.flow_from_directory(test_path,
                                                       target_size = (224,224),
                                                       shuffle = False,
                                                        batch_size = batch_size_res,
                                                       class_mode = 'categorical')


# In[116]:


STEP_SIZE_VALID_res=validation_generator_res.n//validation_generator_res.batch_size
STEP_SIZE_VALID_custom=validation_generator_custom.n//validation_generator_custom.batch_size


# In[106]:


res_model = load_model(path_res_model)
custom_model = load_model(path_custom_model)


# In[107]:


score_res = res_model.evaluate_generator(validation_generator_res, STEP_SIZE_VALID_res)
print('Loss of ResNet: ', score_res[0])
print('Score of ResNet without TTA : ', score_res[1]*100)


# In[117]:


score_custom = custom_model.evaluate_generator(validation_generator_custom, STEP_SIZE_VALID_custom)
print('Loss of Custom Model: ', score_custom[0])
print('Score of Custom Model: ', score_custom[1]*100)


# **TTA of ResNet**

# In[118]:


#True prediction
y_val = validation_generator_custom.classes


# In[119]:


tta_steps = 2
prediction = []

#getting 10 sets of predictions
for i in tqdm.tqdm(range(tta_steps)):
    preds = res_model.predict_generator(validation_generator_res_tta, verbose = 1)
    prediction.append(preds)

#averaging them
preds = np.mean(prediction, axis = 0)

print('Accuracy after TTA in ResNet : ', np.mean(np.equal(y_val, np.argmax(preds,axis = -1))))


# **Ensembling**

# In[120]:


#using prediction from TTA for ResNet
preds_res = np.mean(prediction, axis = 0)
#getting prediction for custom model
preds_custom = custom_model.predict_generator(validation_generator_custom, verbose = 1)


# **Soft Voting Scheme**

# In[121]:


preds_final= (preds_res +  preds_custom)/2.0


# In[122]:


print('Final Accuracy after ensembling : ',np.mean(np.equal(y_val, np.argmax(preds_final,axis = -1))))


# In[ ]:




