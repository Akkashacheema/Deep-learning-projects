#!/usr/bin/env python
# coding: utf-8

# In[35]:


import numpy as np
import os
from sklearn.metrics import confusion_matrix
import seaborn as sn; sn.set(font_scale=1.4)
from sklearn.utils import shuffle           
import matplotlib.pyplot as plt             
import cv2                                 
import tensorflow as tf                
from tqdm import tqdm


# In[36]:


class_names = ['7', '8', '9', '10']
class_names_label = {class_name:i for i, class_name in enumerate(class_names)}

nb_classes = len(class_names)

IMAGE_SIZE = (150, 150)


# In[37]:


def load_data():

    
    datasets = ['C:\\Users\\Mega\\Downloads\\swimming_drowning_dataset\\test', 'C:\\Users\\Mega\\Downloads\\swimming_drowning_dataset\\train']
    output = []
    
    # Iterate through training and test sets
    for dataset in datasets:
        
        images = []
        labels = []
        
        print("Loading {}".format(dataset))
        
        # Iterate through each folder corresponding to a category
        for folder in os.listdir(dataset):
            label = class_names_label[folder]
            
            # Iterate through each image in our folder
            for file in tqdm(os.listdir(os.path.join(dataset, folder))):
                
             
                img_path = os.path.join(os.path.join(dataset, folder), file)
                
  
                image = cv2.imread(img_path)   
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, IMAGE_SIZE) 
                
                # Append the image and its corresponding label to the output
                images.append(image)
                labels.append(label)
                
        images = np.array(images, dtype = 'float32')
        labels = np.array(labels, dtype = 'int32')   
        
        output.append((images, labels))

    return output


# In[38]:


(train_images, train_labels), (test_images, test_labels) = load_data()


# In[39]:


train_images, train_labels = shuffle(train_images, train_labels, random_state=25)


# In[40]:


n_train = train_labels.shape[0]
n_test = test_labels.shape[0]

print ("Number of training examples: {}".format(n_train))
print ("Number of testing examples: {}".format(n_test))
print ("Each image is of size: {}".format(IMAGE_SIZE))


# In[41]:


train_images = train_images / 255.0 
test_images = test_images / 255.0


# In[42]:


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (150, 150, 3)), 
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(6, activation=tf.nn.softmax)
])


# In[43]:


model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])


# In[44]:


history = model.fit(train_images, train_labels, batch_size=128, epochs=20, validation_split = 0.2)


# In[53]:


print(history.history['accuracy'])


# In[54]:


tf.keras.models.save_model(model,'mask_model.h5')


# In[59]:


from tensorflow.keras.preprocessing import image
import numpy as np


# In[61]:


img = image.load_img("C:\\Users\\atifc\\Downloads\\Akasha\\NormalizedData\\test\\7\\Image164.jpg", target_size=(150, 150))
x=image.img_to_array(img) / 255
resized_img_np = np.expand_dims(x,axis=0)
prediction = model.predict(resized_img_np)


# In[62]:


prediction


# In[63]:


print(prediction.argmax())


# In[1]:


import matplotlib.pyplot as plt
accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(accuracy) + 1)
plt.plot(epochs, accuracy, "bo", label="Training accuracy")
plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.figure()
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.show()


# In[ ]:




