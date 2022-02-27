import requests, pickle
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
import numpy as np

import xai_utils
from xai_utils import grad_cam
from shap_s1 import shap_user_defined
from drop_weight import calculate_drop_increase


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
    #print(images.shape)
    # Pre-processing image 
    img=images[0,:,:]
    #img=img_to_array(img)
    img = np.expand_dims(img,axis=0)
    # Getting the prediction for image
    Y=predictions[0][class_index]
    
    grad_array=np.reshape(exmap, (-1,))
    array_size=int(grad_array.shape[0]*frac)
    thr=np.flip(sorted(grad_array))[array_size]
    exmap1_msk=(exmap>thr)
    exmap1_thr=np.zeros(shape=(1,40,1))
    exmap1_thr=img*exmap1_msk
    ex_predictions = model.predict(exmap1_thr)[0]
    O1=ex_predictions[class_index]
    etta=(Y-O1)/(Y+1e-100)
    return (etta*(etta>0), 1*(etta<0), Y, O1)

with open('./MNIST1D.pkl', 'rb') as handle:
    dataset = pickle.load(handle)
weight_decay = 5e-4 
#print(dataset.keys())

x_train = dataset['x']    
y_train = dataset['y']
x_test = dataset['x_test']
y_test = dataset['y_test'] 


model = Sequential()
#print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
x_train = np.expand_dims(x_train,2)
x_test = np.expand_dims(x_test,2)

one_hot_y = np.zeros((y_train.size,10))
one_hot_y[np.arange(y_train.size),y_train]=1
y_train = one_hot_y

one_hot_y_test = np.zeros((y_test.size,10))
one_hot_y_test[np.arange(y_test.size),y_test]=1
y_test = one_hot_y


model.add(layers.Conv1D(25,5,1,'same',activation='relu',kernel_regularizer=regularizers.l2(weight_decay),bias_regularizer=regularizers.l2(weight_decay)))
model.add(layers.Conv1D(25,3,1,'same',activation='relu',kernel_regularizer=regularizers.l2(weight_decay),bias_regularizer=regularizers.l2(weight_decay)))
model.add(layers.Conv1D(25,3,1,'same',activation='relu',kernel_regularizer=regularizers.l2(weight_decay),bias_regularizer=regularizers.l2(weight_decay)))
model.add(layers.Flatten())
model.add(layers.Dense(10,activation='softmax',kernel_regularizer=regularizers.l2(weight_decay),bias_regularizer=regularizers.l2(weight_decay)))


loss_fn = keras.losses.CategoricalCrossentropy()
model.compile(optimizer='sgd',loss=loss_fn,metrics=['accuracy'])

logdir = "training_result"
tensorboard_callback = keras.callbacks.TensorBoard(logdir,histogram_freq=1)
history = model.fit(x_train,y_train,epochs=200, callbacks=tensorboard_callback)
#history = model.fit(x_train,y_train,epochs=1, callbacks=tensorboard_callback)
model.summary()


#print(x_train.shape)      #(4000,40,1)
#print(x_test.shape)       #(1000,40,1)

indx = 870

#print(x_test[0:1].shape)
#print(np.expand_dims(x_test[0:1],2).shape)


#print(shap_values)
#shap.image_plot(shap_values, -x_test)
for idx in range(10):
    # plot test data
    plt.subplot(1,3,1)
    str1 = "input array : digit" + str(dataset['y_test'][idx+indx])
    plt.plot(x_test[idx+indx])
    plt.title(str1)

    predict_y=model.predict(x_test[indx+idx:indx+idx+1])
    class_indx = np.argmax(predict_y[0])
    shapley=shap_user_defined(x_test[indx+idx:idx+indx+1],model,[2,1],500)

    total_drop = calculate_drop_increase(x_test[indx+idx:indx+idx+1], model, shapley, class_indx, 0.3)

    plt.subplot(1,3,2)
    plt.plot(abs(shapley))
    plt.title('Shapley value')

    x2 = x_test[idx+indx].copy()
    plt.subplot(1,3,3)
    plt.plot(x2,color='blue')
    x2[abs(shapley)<0.05]=np.nan
    plt.plot(x2,color = 'red')
    plt.title('Highlighted input region')
    plt.show()
print(total_drop)

