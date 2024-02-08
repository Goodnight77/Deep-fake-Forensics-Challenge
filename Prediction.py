import os
import argparse
import random
import numpy as np
import tensorflow as tf
from PIL import Image  # Assuming you want to use Image module to save the image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K  # Added this import for Keras backend

smooth=1
def dice_coef(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred)
    return (2.0 * intersection + smooth) / (union + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return dice_coef_loss(y_true, y_pred) + bce(y_true, y_pred)
class PyramidPoolingModule(tf.keras.layers.Layer):
    def __init__(self, num_filters=1, kernel_size=(1, 1), bin_sizes=[1, 2, 3, 6], pool_mode='avg', **kwargs):
        super(PyramidPoolingModule, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.bin_sizes = bin_sizes
        self.pool_mode = pool_mode
        self.pyramid_pooling = self.build_pyramid_pooling()

    def build_pyramid_pooling(self):
        return PyramidPoolingModule.PyramidPoolingModule(
            num_filters=self.num_filters,
            kernel_size=self.kernel_size,
            bin_sizes=self.bin_sizes,
            pool_mode=self.pool_mode,
        )

    def call(self, inputs):
        return self.pyramid_pooling(inputs)

    class PyramidPoolingModule(tf.keras.layers.Layer):
        def __init__(self, num_filters, kernel_size, bin_sizes, pool_mode, **kwargs):
            super(PyramidPoolingModule.PyramidPoolingModule, self).__init__(**kwargs)
            self.num_filters = num_filters
            self.kernel_size = kernel_size
            self.bin_sizes = bin_sizes
            self.pool_mode = pool_mode
            self.pyramid_layers = []

            for bin_size in bin_sizes:
                self.pyramid_layers.append(
                    layers.Conv2D(
                        filters=num_filters,
                        kernel_size=kernel_size,
                        padding='same',
                        activation='relu'
                    )
                )

        def call(self, inputs):
            outputs = [inputs]

            for i, bin_size in enumerate(self.bin_sizes):
                pooled = tf.keras.layers.AveragePooling2D(pool_size=(bin_size, bin_size))(inputs)
                convolved = self.pyramid_layers[i](pooled)
                resized = tf.image.resize(convolved, tf.shape(inputs)[1:3])
                outputs.append(resized)

            return tf.concat(outputs, axis=-1)

class BayarConv2d(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=0):
        super(BayarConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.minus1 = tf.ones((self.in_channels, self.out_channels, 1)) * -1.000

        # only (kernel_size ** 2 - 1) trainable params as the center element is always -1
        self.kernel = self.add_weight(shape=(self.in_channels, self.out_channels, kernel_size ** 2 - 1),
                                      initializer='random_normal',
                                      trainable=True)

    def bayarConstraint(self):
        kernel_permuted = tf.transpose(self.kernel, perm=[2, 0, 1])
        kernel_sum = tf.reduce_sum(kernel_permuted, axis=0)
        ctr = self.kernel_size ** 2 // 2
        real_kernel = tf.concat([self.kernel[:, :, :ctr], self.minus1, self.kernel[:, :, ctr:]], axis=2)
        real_kernel = tf.reshape(real_kernel, (self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        return real_kernel

    def call(self, x):
        x = tf.nn.conv2d(x, self.bayarConstraint(), strides=self.stride, padding='SAME')
        return x


# Define the command line arguments
parser = argparse.ArgumentParser(description="Generate predictions for a random image in the specified folder.")
parser.add_argument("--images_folder", type=str, required=True, help="Path to the sample images folder.")
args = parser.parse_args()

# Load the pre-trained model
model_path = './saved_model/best_model.h5'  # Replace with the actual path to your pre-trained model
from tensorflow.keras.models import load_model
import keras
keras.utils.get_custom_objects()['dice_coef'] = dice_coef
keras.utils.get_custom_objects()['dice_coef_loss'] = dice_coef_loss
keras.utils.get_custom_objects()['BayarConv2d'] = BayarConv2d
keras.utils.get_custom_objects()['PyramidPoolingModule'] = PyramidPoolingModule


model=load_model(model_path,compile=False)

image_files = [f for f in os.listdir(args.images_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

random_image = random.choice(image_files)
image_path = os.path.join(args.images_folder, random_image)

image = load_img(image_path, target_size=(256, 256))
image_array = img_to_array(image) / 255.0
image_array = np.expand_dims(image_array, axis=0)
prediction, classif = model.predict(image_array)
predicted_mask = prediction[0]
if classif>0.5:
    print(f'{random_image} is manipulated!')
else:
    print(f'{random_image} is real!')
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(img_to_array(load_img(image_path, target_size=(256, 256)))/255)
plt.title('Original Image')


plt.subplot(1, 2, 2)
plt.imshow(predicted_mask*(classif>0.5), cmap='gray')
plt.title('Predicted Mask')

plt.show()

predicted_mask = predicted_mask * 255
predicted_mask = predicted_mask.astype(np.uint8)

predicted_mask_image = Image.fromarray(predicted_mask.squeeze(), mode='L')

save_path = '/predicted_mask.png' 
predicted_mask_image.save(save_path)
