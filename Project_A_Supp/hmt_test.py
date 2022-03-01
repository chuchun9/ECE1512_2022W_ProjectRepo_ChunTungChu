import os

import numpy as np
import pandas as pd
import tensorflow 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Flatten
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from tensorflow.core.util import event_pb2
from tensorflow.python.lib.io import tf_record
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import cv2

#SHAP
import xgboost
import shap

from matplotlib import test
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

import xai_utils
from xai_utils import *

model=tf.keras.models.load_model('models/HMT.h5')

train_dir = 'hmt_dataset/HMT_train' #you should change to your directory
test_dir = 'hmt_dataset/HMT_test' #you should change to your directory

train_datagen = ImageDataGenerator(rescale=1/255.,
shear_range=0.1,
rotation_range=15,
horizontal_flip=True,
vertical_flip=True)

test_datagen = ImageDataGenerator(rescale=1/255.)

train_generator = train_datagen.flow_from_directory(train_dir,
class_mode='categorical',
interpolation='bilinear',
target_size=(224, 224),
batch_size=32,
shuffle=True)

test_generator = test_datagen.flow_from_directory(test_dir,
class_mode='categorical',
interpolation='bilinear',
target_size=(224, 224),
batch_size=32,
shuffle=False)

def calculate_drop_increase(images, model, exmap, class_index, frac=0.15):
    '''
    inputs:
        images: a 4-D image of size (1 x H x W x 3)
          containing an image in RGB format and of size (H x W)
        model: The base model
        exmap: a given explanation map whose completeness is to be evaluated.
        class_index: The class to whom the explanation map is related to.
        frac: The fraction of top pixels selected.
    returns:v
        a tuple with 4 calculates values:
        (drop, increase, original_pred, eplanation_pred)
        drop (float): drop rate (between 0 and 1)
        increase (boolean): "1" if increase happened
        original_pred: confidence score for original image
        explanation_pred:  confidence score for the selected top pixels of the image.
    '''
    predictions = model.predict(images)
    # Pre-processing image 
    img=images[0,:,:,:]
    img=img_to_array(img)
    img = np.expand_dims(img,axis=0)
    # Getting the prediction for image
    Y=predictions[0][class_index]
    
    grad_array=np.reshape(exmap, (-1,))
    array_size=int(grad_array.shape[0]*frac)
    thr=np.flip(sorted(grad_array))[array_size]
    exmap1_msk=(exmap>thr)
    exmap1_thr=np.zeros(shape=(1,224,224,3))
    exmap1_thr[0,:,:,0]=img[0,:,:,0]*exmap1_msk
    exmap1_thr[0,:,:,1]=img[0,:,:,1]*exmap1_msk
    exmap1_thr[0,:,:,2]=img[0,:,:,2]*exmap1_msk
    ex_predictions = model.predict(exmap1_thr)[0]
    O1=ex_predictions[class_index]
    etta=(Y-O1)/(Y+1e-100)
    return (etta*(etta>0), 1*(etta<0), Y, O1)

def grad_cam(input_model, image, layer_name):
    cls = np.argmax(input_model.predict(image))
    def normalize(x):
        """Utility function to normalize a tensor by its L2 norm"""
        return (x + 1e-10) / (K.sqrt(K.mean(K.square(x))) + 1e-10)
    """GradCAM method for visualizing input saliency."""
    y_c = input_model.output
    conv_output = input_model.get_layer(layer_name).output
    feedforward1 = tensorflow.keras.models.Model([input_model.input], [conv_output, y_c])
    with tf.GradientTape() as tape:
        ff_results=feedforward1([image])
        all_fmap_masks, predictions = ff_results[0], ff_results[-1]
        loss = predictions[:, cls]
    grads_val = tape.gradient(loss, all_fmap_masks)
    if len(image.shape)==3:
        axis=(0, 1)
    elif len(image.shape)==4:
        axis=(0, 1, 2)
    weights = np.mean(grads_val, axis=axis)
    cam = np.dot(all_fmap_masks[0], weights)
    #print (cam)
    H,W= image.shape[1:3]
    cam = np.maximum(cam, 0)
    #cam = resize(cam, (H, W))
    cam = zoom(cam,H/cam.shape[0])
    #cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    return cam


def smooth_grad(input_model, image, n, noise_level):
  cls = np.argmax(input_model.predict(image))
  smooth_grads = np.zeros_like(image)
  std = np.sqrt(noise_level * (np.max(image) - np.min(image)))
  feedforward = tensorflow.keras.models.Model(input_model.input, input_model.output)
  i = 0
  while i < n:
    noise_input = image + np.random.normal(0, std, image.shape)
    noise_input = tf.Variable(noise_input, dtype=tf.float32)
    # noise_input = image
    with tf.GradientTape() as tape:
      preds = feedforward([noise_input])
      noise_label = np.argmax(preds)
      loss = preds[:, cls]
    if noise_label == cls:
      grads = tape.gradient(loss, noise_input)
      smooth_grads += np.abs(grads)
      # smooth_grads += grads
    i += 1
  smooth_grads /= i
  if len(smooth_grads.shape) == 4:
    smooth_grads = np.mean(smooth_grads, axis=3)[0]
  return smooth_grads

  ############

## overall classification accuracy on test set
#test_loss, test_acc = model.evaluate_generator(generator=test_generator)
#print("Overall Accuracy is", test_acc)

#prob = model.predict_generator(test_generator)
#y_pred = np.array([np.argmax(x) for x in prob])
#y_test = test_generator.classes
#cm = confusion_matrix(y_test, y_pred, normalize='true')
# cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#class_acc = cm.diagonal()
#one_hot_y_test = np.zeros((y_test.shape[0], 8))
#one_hot_y_test[np.arange(y_test.size),y_test] = 1

classes=['Tumor', 'Stroma', 'Complex', 'Lympho', 'Debris', 'Mucosa', 'Adiopse', 'Empty']
print(test_generator.class_indices)
#print('accuracy of each class: \n')
#for i in range(len(classes)):
#  print(classes[i], ' : ', class_acc[i])
#print('\n')

## Randomly electing background for shap explainer
background = []
cur_num = 0
flag = True
train_generator.reset()
a = 20
while cur_num != a:
  image_batch, label_batch = train_generator.next()
  if a - cur_num < image_batch.shape[0]:
    background.append(image_batch[np.random.choice(image_batch.shape[0], a - cur_num, replace=False)])
    cur_num = a
  else:
    background.append(image_batch)
    cur_num += image_batch.shape[0]

background = np.concatenate(background, axis=0)
shap_explainer = shap.DeepExplainer(model, background)
print(background.shape)

#### pick one image that's classified correctly from each category

test_generator.reset()
each_cat = []
each_lbl = []
found = set()
flag = True
while flag:
  image_batch, label_batch = test_generator.next()
  prediction = model(image_batch)
  prediction = np.argmax(prediction, axis=1)
  label_batch = np.argmax(label_batch, axis=1)
  #print(prediction)
  #print(label_batch)
  for i in range(len(prediction)):
    if len(found) == len(classes):
      flag = False
    if label_batch[i] not in found:
      if prediction[i] == label_batch[i]:
        found.add(label_batch[i])
        each_cat.append(image_batch[i])
        each_lbl.append(label_batch[i])

each_cat = np.array(each_cat)
print(each_cat.shape)
print(each_lbl)

# starting using explanation algorithms
for i in range(each_cat.shape[0]):
  #class_index = np.argmax(prediction[0])
  #e = shap.DeepExplainer(model, background)
  class_index = each_lbl[i]
  print(class_index)
  print(classes)
  shapley = shap_explainer.shap_values(np.expand_dims(each_cat[i], axis=0))
  #shap.image_plot(shapley,each_cat[i])
  
  explanation_map_SHAP3 = np.array(shapley)
  explanation_map_SHAP3 = explanation_map_SHAP3[class_index,0]
  explanation_map_SHAP = np.zeros([224,224])
  explanation_map_SHAP[:,:] = (explanation_map_SHAP3[:,:,1]+explanation_map_SHAP3[:,:,2]+explanation_map_SHAP3[:,:,0])/3
  
  #explanation_map_SHAP = shap_user_defined(np.expand_dims(image_batch[index], axis=0), model, [7,7],1)
  explanation_map_SHAP = explanation_map_SHAP
  explanation_map_SHAP -= explanation_map_SHAP.min()
  explanation_map_SHAP = explanation_map_SHAP / (explanation_map_SHAP.max()+10e-30)

  explanation_map_GradCAM = grad_cam(model, np.expand_dims(each_cat[i], axis=0), 'max_pooling2d_1')
  explanation_map_GradCAM -= explanation_map_GradCAM.min()
  explanation_map_GradCAM /= explanation_map_GradCAM.max()+10e-30

  explanation_smooth_grad = smooth_grad(model, np.expand_dims(each_cat[i], axis=0), 50, 0.05)
  explanation_smooth_grad -= explanation_smooth_grad.min()
  explanation_smooth_grad /= explanation_smooth_grad.max()+10e-30

  # explanation_shap = shap_explainer.shap_values(np.expand_dims(each_cat[i], axis=0))

  plt.figure(figsize=(20,5))

  plt.subplot(1,4,1)
  plt.imshow(each_cat[i])
  plt.axis('off')
  plt.title('Sample image ({})'.format(classes[each_lbl[i]]))

  plt.subplot(1,4,2)
  plt.imshow(each_cat[i])
  plt.imshow(explanation_map_GradCAM, cmap='jet', alpha=0.5)
  plt.axis('off')
  plt.title('Explanation map (Grad-CAM)')

  plt.subplot(1,4,3)
  plt.imshow(each_cat[i])
  plt.imshow(explanation_smooth_grad, cmap='jet', alpha=0.5)
  plt.axis('off')
  plt.title('Explanation map (Smooth Grad)')

  plt.subplot(1,4,4)
  plt.imshow(each_cat[i])
  plt.imshow(explanation_map_SHAP, cmap='jet', alpha=0.5)
  plt.axis('off')
  plt.title('Explanation map (SHAP)')
  plt.show()

  plt.savefig("{}.png".format(i))



test_generator.reset()

all_sample_classes = test_generator.classes
drop_rate = 0.
increase_rate = 0
cnt = 0
for _ in range(15):
    image_batch,label_batch=test_generator.next()
    prediction=model(image_batch)
    for index in range(32):
        shapley = shap_explainer.shap_values(np.expand_dims(image_batch[index], axis=0))
        #shap.image_plot(shapley,each_cat[i])
  
        explanation_map_SHAP3 = np.array(shapley)
        explanation_map_SHAP3 = explanation_map_SHAP3[np.argmax(prediction[index]),0]
        explanation_map_SHAP = np.zeros([224,224])
        explanation_map_SHAP[:,:] = (explanation_map_SHAP3[:,:,1]+explanation_map_SHAP3[:,:,2]+explanation_map_SHAP3[:,:,0])/3
   
        res = calculate_drop_increase(np.expand_dims(image_batch[index], axis=0), model, explanation_map_SHAP, class_index=np.argmax(prediction[index]), frac=0.9)
        drop_rate += res[0]
        increase_rate += res[1]
        print(drop_rate)
        print(increase_rate)
        print(cnt)
        cnt = cnt+1
drop_rate /= (15*32)
increase_rate /= (15*32)
print(drop_rate)
print(increase_rate)
