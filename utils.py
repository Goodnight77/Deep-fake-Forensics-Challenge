
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
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


