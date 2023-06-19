#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras import  layers, models
   


# In[2]:


image_dir = Path(r"C:\Users\DELL\Desktop\Scalogram")
filePaths= pd.Series(list(image_dir.glob(r'**/*.png')), name= 'FilePaths').astype(str)
os.path.split(filePaths[1720])[0].split('_')[1]


# In[3]:


labels = filePaths.apply(lambda x: os.path.split(x)[0].split('_')[1]).astype(np.int16)
labels


# In[4]:


ImgTypeIndex = pd.Series(filePaths.apply(lambda x: os.path.split(x)[0].split('_')[1]).astype(np.int16), name='ImgTypeIndex')
filePaths.shape


# In[5]:


imgs_=[]
for img in filePaths :
    with Image.open(img) as test_image:
        imgs_.append(np.asarray(test_image))
        #test_image.close()
data=[]
for im in imgs_:
    data.append(im)


# In[6]:


classes = ["Blink","Horizontal Left","Horizontal Right","Vertical  Down", "Vertical  Up"]


# In[7]:


from sklearn.model_selection import train_test_split
x_train , x_test = train_test_split( imgs_, train_size=0.8, shuffle=True, random_state=1)
y_train , y_test = train_test_split( ImgTypeIndex.values, train_size=0.8, shuffle=True, random_state=1)


# In[8]:


def plot_sample(X, y, index):
    #img = Image.open(X[index])
    plt.figure(figsize = (15,2))
    plt.imshow(X[index])
    plt.xlabel(classes[int(y[index])])


# In[9]:


x_train[600].shape


# In[10]:


plot_sample(x_train,y_train , 10)


# In[11]:


x_train = np.asarray(x_train) / 255.0


# In[12]:


x_test = np.asarray(x_test) / 255.0


# In[13]:


cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(369, 496, 4)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(5, activation='softmax')
])


# In[14]:


cnn.compile(optimizer='SGD',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[15]:


cnn.fit(x_train, y_train, epochs=100)


# In[16]:


loss, accuracy = cnn.evaluate(x_test, y_test)


# In[17]:


import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
# Predict on test data
y_pred = cnn.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
# Compute confusion matrix
confusion_mtx = confusion_matrix(y_test, y_pred_classes)
# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mtx, annot=True, fmt="d", xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# In[18]:


from sklearn.metrics import precision_score

# Generate predictions for the test data
y_pred = cnn.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Get the true labels for the test data
y_true = y_test

# Calculate the precision score for each class
precision_scores = precision_score(y_true, y_pred_classes, average=None)



# Plot the precision score for each class
plt.bar(classes, precision_scores)
plt.title('Precision Score')
plt.xlabel('Class')
plt.ylabel('Precision')
plt.show()


# In[19]:


cnn.save(r"C:\\Users\\DELL\\Desktop\\Scalogram\\model_result")


# In[21]:


import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model('C:\\Users\\DELL\\Desktop\\Scalogram\\model_result')

# Load and preprocess the image
image = Image.open(r"C:\Users\DELL\Desktop\Scalogram\Vertical  Down_3\EOGVD 704_Scalogram.png")
image = np.expand_dims(image, axis=0)  # Add batch dimension
# Make predictions
predictions =model.predict(np.asarray(image)/255.0)
predicted_class = np.argmax(predictions, axis=1)
print(predicted_class)
class_name = classes[predicted_class[0]]
print(class_name)


# In[ ]:




