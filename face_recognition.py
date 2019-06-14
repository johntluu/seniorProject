#!/cm/shared/apps/virtualenv/csc/bin/python3.6
#%%
from datasetLoader import load_all_datasets
import os
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow.keras as keras
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import MaxPooling3D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.layers import Input, Dense, Dropout, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

#%% [markdown]
# ### Senior Project: Face Recognition Using Generated Depth Information and 2D Images
# 
# This project uses the Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network (ECCV 2018) to generate the depth information from 2D images from the VGGFace2 dataset. Both the 2D image and the depth information is then fed into the network
#%% [markdown]
# ### Step 1: Load the dataset.
# Uses the python loader script to load all the images into the notebook.

#%%
x_train, y_train, x_validation, y_validation, x_test, y_test, depth_train, depth_validation, depth_test = load_all_datasets("./dataset5/")

# print(np.count_nonzero(x_train))
# x_train = (x_train * 2) - 1
# x_val = (x_val * 2) - 1
# x_test = (x_test * 2) - 1

#%%
print(len(depth_train[0]))
#%%
# Try showing images
print(len(x_train))
print(len(x_validation))
print(len(x_test))
plt.figure(figsize = (20, 10))

columnsTrain = len(x_train)
columnsValidation = len(x_validation)
columnsTest = len(x_test)

# for i in range(columnsTrain):
#     plt.subplot(5 / columnsTrain + 1, columnsTrain, i + 1)
#     plt.imshow(cv2.cvtColor(x_train[i], cv2.COLOR_BGR2RGB))

# for i in range(columnsValidation):
#     plt.subplot(5 / columnsValidation + 1, columnsValidation, i + 1)
#     plt.imshow(cv2.cvtColor(x_validation[i], cv2.COLOR_BGR2RGB))

for i in range(columnsTest):
    plt.subplot(5 / columnsTest + 1, columnsTest, i + 1)
    plt.imshow(cv2.cvtColor(x_test[i], cv2.COLOR_BGR2RGB))


#%% [markdown]
# ### Step 2: Set up a VGG-style neural network for multi-class classification.
# 
# Since the input of the neural network is 3D, we have to use 3D convolutional layers as well as 3D max pooling. To fit the VGG-style, we have 3 layers of convolutions before each max pooling layer. Then add the flatten, dense, relu activation, dense, and softmax activation layers. The input shape is 32x32x32 because if it was any larger, TensorFlow says it would allocate too much memory from the system.


#%%
# RGB CNN

inputShapeRGB = (128, 128, 3)
inputRGB = Input(shape = inputShapeRGB)
chanDim = -1
regress = False

filters = (16, 32, 64)

for (i, f) in enumerate(filters):
    print("i: {} f: {}".format(i, f))
    if (i == 0):
        x = inputRGB
    
    x = Conv2D(f, (3, 3), padding = "same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis = chanDim)(x) # normalize each batch by both mean and variance reference
    x = MaxPooling2D(pool_size = (2, 2))(x)

x = Flatten()(x)
x = Dense(16)(x)
x = Activation("relu")(x)
x = BatchNormalization(axis = chanDim)(x)
x = Dropout(0.5)(x) # helps prevent overfitting by randomly setting a fraction 0.5 of input units to 0
x = Dense(4)(x)
x = Activation("relu")(x)

if (regress):
    x = Dense(1, activation = "linear")(x)

rgbModel = Model(inputRGB, x)
print(rgbModel.summary())



#%%
# Depth CNN

#inputShapeDepth = (1000, 1000, 1)
inputShapeDepth = (1500, 1500, 1)
inputDepth = Input(shape = inputShapeDepth)

for (i, f) in enumerate(filters):
    print("i: {} f: {}".format(i, f))
    if (i == 0):
        y = inputDepth
    
    y = Conv2D(f, (3, 3), padding = "same")(y)
    y = Activation("relu")(y)
    y = BatchNormalization(axis = chanDim)(y) # normalize each batch by both mean and variance reference
    y = MaxPooling2D(pool_size = (2, 2))(y)

y = GlobalMaxPooling2D()(y)
y = Dense(16)(y)
y = Activation("relu")(y)
y = BatchNormalization(axis = chanDim)(y)
y = Dropout(0.5)(y) # helps prevent overfitting by randomly setting a fraction 0.5 of input units to 0
y = Dense(4)(y)
y = Activation("relu")(y)

if (regress):
    y = Dense(1, activation = "linear")(y)

depthModel = Model(inputDepth, y)
print(depthModel.summary())

#%%
# Combine the two CNNs together
combined = concatenate([rgbModel.output, depthModel.output])
final = Dense(4, activation = "relu")(combined)
# final = Dense(3, activation = "softmax")(final) # 3 temp, but use # of identities
final = Dense((max(y_train) + 1), activation = "softmax")(final)
finalModel = Model(inputs = [rgbModel.input, depthModel.input], outputs = final)
print(finalModel.summary())

#%% [markdown]
# ### Step 4: Train the model on the training data.
# This is where we compile the model with a specific loss function. We also expand the dimensions of the training, test, and validation sets to fit the model when we train it.

#%%
opt = Adam(lr = 1e-2, decay = 1e-2 / 200)
finalModel.compile(loss = 'sparse_categorical_crossentropy',
              optimizer = opt,
              metrics = ['categorical_accuracy'])


depthTrainingImages = []
depthValidationImages = []
depthTestingImages = []

for i in range(len(depth_train)):
    dt = np.zeros((1500, 1500, 1))
    for j in range(len(depth_train[i])):
        for k in range(len(depth_train[i][j])):
            dt[j][k] = depth_train[i][j][k]
    depthTrainingImages.append(dt)

for i in range(len(depth_validation)):
    dt = np.zeros((1500, 1500, 1))
    for j in range(len(depth_validation[i])):
        for k in range(len(depth_validation[i][j])):
            dt[j][k] = depth_validation[i][j][k]
    depthValidationImages.append(dt)

for i in range(len(depth_test)):
    dt = np.zeros((1500, 1500, 1))
    for j in range(len(depth_test[i])):
        for k in range(len(depth_test[i][j])):
            dt[j][k] = depth_test[i][j][k]
    depthTestingImages.append(dt)

# batch size 2 works
history = finalModel.fit([np.array(x_train), np.asarray(depthTrainingImages)], y_train,
                    batch_size = 2,
                    epochs = 200,
                    verbose = 0,
                    validation_data=([np.array(x_validation), np.asarray(depthValidationImages)], y_validation))

# history = finalModel.fit([np.array(x_train), np.asarray(depthTrainingImages)], [0] * len(y_train),
#                     batch_size = 8,
#                     epochs = 200,
#                     verbose = 1,
#                     validation_data=([np.array(x_validation), np.asarray(depthValidationImages)], [0] * len(y_validation)))



#%% [markdown]
# ### Step 5: Loss and Accuracy Plots
# This graph shows the loss and accuracy throughout training.

#%%
'''
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training Loss','Testing Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.legend(['Training Accuracy','Testing Accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()
'''
#%% [markdown]
# ### Step 6: Accuracy on the training, validation and testing data.
# This is the final loss and final accuracy found through this method.

#%%
evaluation = finalModel.evaluate(x = [np.array(x_test), np.asarray(depthTestingImages)], y = y_test, batch_size = 2)
print("Loss")
print(history.history["loss"])
print("Val Loss")
print(history.history["val_loss"])
print("Categorical Accuracy")
print(history.history["categorical_accuracy"])
print("Val Categorical Accuracy")
print(history.history["val_categorical_accuracy"])
print(finalModel.metrics_names)
print(evaluation)
