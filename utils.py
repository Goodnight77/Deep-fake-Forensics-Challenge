
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras import backend as K
from tensorflow import keras
from tensorflow.keras import layers

#this is the set of learnable filters 
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
        
def add_random_boxes(img,n_k,size=32):
    h,w = size,size
    img = np.asarray(img)
    img_size = img.shape[1]
    boxes = []
    for k in range(n_k):
        y,x = np.random.randint(0,img_size-w,(2,))
        img[y:y+h,x:x+w] = 0
        boxes.append((x,y,h,w))
    return img
class DataGeneratorA(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size=32, shuffle=True, augment=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.on_epoch_end()
        self.Image_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            brightness_range=[0.2, 1.8]
        )
        self.iimage_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            brightness_range=[3, 4]
        )
        if self.augment:
            self.image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=45,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest',
            )

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        batch_data = self.data[index * self.batch_size: (index + 1) * self.batch_size]

        images = []
        masks = []
        labels = []

        for _, row in batch_data.iterrows():
            image = load_img(row['image'], target_size=(256, 256))
            image = img_to_array(image)
            if self.augment==True:
                image = self.Image_gen.random_transform(image)
            edge = load_img(row['edge'], target_size=(256, 256))
            edge = img_to_array(edge)
            edge/=255.0

            mask = load_img(row['mask'], target_size=(256, 256))
            image/=255.0
            mask = img_to_array(mask) / 255.0
            modified=image*edge
            aug=self.iimage_gen.random_transform(modified*255.0)
            final=image-modified+aug/255.0
            augmented = add_random_boxes(np.concatenate([final, mask], axis=-1),3,70)
            final, mask = np.split(augmented, 2, axis=-1)



            if self.augment:
                augmented = self.image_datagen.random_transform(np.concatenate([final, mask], axis=-1))
                final, mask = np.split(augmented, 2, axis=-1)
                mask=tf.image.rgb_to_grayscale(mask)

            images.append(final)
            masks.append(mask)
            labels.append(row['class'])

        images = np.array(images)
        masks = np.array(masks)
        labels = np.array(labels)

        return images, [masks, labels]

    def on_epoch_end(self):
        if self.shuffle:
            self.data = self.data.sample(frac=1)


class DataGeneratorNA(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size=32, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        batch_data = self.data[index * self.batch_size : (index + 1) * self.batch_size]

        images = []
        masks = []
        labels = []

        for _, row in batch_data.iterrows():
            image = load_img(row['image'], target_size=(256, 256))
            image = img_to_array(image) / 255.0
            images.append(image)

            mask = load_img(row['mask'], target_size=(256, 256), color_mode='grayscale')
            mask = img_to_array(mask) / 255.0
            masks.append(mask)

            labels.append(row['class'])

        images = np.array(images)
        masks = np.array(masks)
        labels = np.array(labels)

        return images, [masks, labels]

    def on_epoch_end(self):
        if self.shuffle:
            self.data = self.data.sample(frac=1)


