# This is a colab code
# Download the dataset
#!wget --no-check-certificate \
#  https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip \
#  -O /tmp/cats_and_dogs_filtered.zip

import os
import zipfile
# Unzip data
local_zip = '/tmp/cats_and_dogs_filtered.zip'
zip_ref = zipfile.ZipFile(local_zip,'r')
zip_ref.extractall('/tmp')
zip_ref.close()

# Define directories
base_dir = '/tmp/cats_and_dogs_filtered'
train_dir = os.path.join(base_dir,'train')
validation_dir = os.path.join(base_dir,'validation')
# Directory with training pictures
train_cats_dir = os.path.join(train_dir,'cats')
train_dogs_dir = os.path.join(train_dir,'dogs')
# Directory with validation pictures
validation_cats_dir = os.path.join(validation_dir,'cats')
validation_dogs_dir = os.path.join(validation_dir,'dogs')

# Show some filenames
train_cat_fnames = os.listdir(train_cats_dir)
train_dog_fnames = os.listdir(train_dogs_dir)
print(train_cat_fnames[:10])
print(train_dog_fnames[:10])

# See the number of images in different directories
print('total training cat images: ',len(os.listdir(train_cats_dir)))
print('total training dog images: ',len(os.listdir(train_dogs_dir)))
print('total validation cat images: ',len(os.listdir(validation_cats_dir)))
print('total validation dog images: ',len(os.listdir(validation_dogs_dir)))

# Show some sample images, configure matplot
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# Output images in 4x4 configuration
nrows = 4
ncols = 4
# Index for iterating over images
pic_index = 0
# Display 8 cat and 8 dog pictures
fig = plt.gcf()
fig.set_size_inches(ncols*4,nrows*4)
pic_index += 8

next_cat_pix = [os.path.join(train_cats_dir,fname)
                for fname in train_cat_fnames[ pic_index-8:pic_index ]
                ]
next_dog_pix = [os.path.join(train_dogs_dir,fname)
                for fname in train_dog_fnames[ pic_index-8:pic_index ]
                ]

for i,img_path in enumerate(next_cat_pix + next_dog_pix):
    sp = plt.subplot(nrows,ncols,i+1)
    sp.axis('Off')
    img = mpimg.imread(img_path)
    plt.imshow(img)

plt.show()

# Define our network
import tensorflow as tf

model = tf.keras.models.Sequential([
    # 1st convolution
    tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(15,150,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    # 2nd convolution
    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # 3rd convolution
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten layer
    tf.keras.layers.Flatten(),
    # Hidden layer
    tf.keras.layers.Dense(512,activation='relu'),
    # Output layer
    tf.keras.layers.Dense(1,activation='sigmoid')
])

# See the transformation of images
model.summary()

# Configuration for training
from tensorflow.keras.optimizers import RMSprop

model.compile(optimizer = RMSprop(lr=0.001),
              loss = 'binary_crossentropy',
              metrics = ['acc'])

# Data preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images rescaled by 1/255
train_datagen = ImageDataGenerator( rescale = 1.0/255. )
test_datagen = ImageDataGenerator( rescale = 1.0/255.)

# -----------
# Flow training images in batches of 20 using train_datagen generator
# -----------
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    target_size=(150,150))
# -----------
# Flow validation images in batches of 20 using test_datagen generator
# -----------
validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        batch_size=20,
                                                        class_mode='binary',
                                                        target_size=(150,150))

# Training
history = model.fit_generator(train_generator,
                              validation_data=validation_generator,
                              steps_per_epoch=100,
                              epochs=15,
                              validation_steps=50,
                              verbose=2)

# Running the model
import numpy as np

from google.colab import files
from keras.preprocessing import image

uploaded = files.upload()

for fn in uploaded.keys():
    # predict images
    path = '/content/' + fn
    img = image.load_img(path,target_size=(150,150))

    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0)
    images = np.vstack([x])
    classes = model.predict(images,batch_size=10)

    print(classes[0])
    if classes[0] > 0:
        print(fn + " is a dog")
    else:
        print(fn + " is a cat")

# Evaluating accuracy and loss
# -----------
# Retrieve a list of results on training and test data
# Sets for each epoch
# -----------
acc      = history.history[      'acc' ]
val_acc  = history.history[  'val_acc' ]
loss     = history.history[     'loss' ]
val_loss = history.history[ 'val_loss' ]

# Get number of epochs
epochs = range(len(acc))

# -----------
# Plot training and validation accuracy per epoch
# -----------
plt.plot  ( epochs,     acc)
plt.plot  ( epochs,val_acc )
plt.title ('Training and validation accuracy')
plt.figure()

# -----------
# Plot training and validation loss per epoch
# -----------
plt.plot  ( epochs,     loss)
plt.plot  ( epochs,val_loss )
plt.title ('Training and validation loss')

# Clean up
import os,signal

os.kill(os.getpid(),
        signal.SIGKILL)