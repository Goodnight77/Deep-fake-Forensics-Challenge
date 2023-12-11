from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import  concatenate
from tensorflow import keras
import tensorflow as tf
# Pyramid Pooling Module
from tensorflow.keras import layers, models
import numpy as np
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
def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    """
    3x3 convolution with padding
    """
    return tf.keras.layers.Conv2D(
        filters=out_planes,
        kernel_size=(3, 3),
        strides=stride,
        padding='same',
        use_bias=has_bias,
        activation=None
    )

def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

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

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_encoder(input_shape=(256, 256, 3)):
    # Define the input layer
    inputs = Input(input_shape)

    # Load pre-trained ResNet50 model
    resnet50 = ResNet50(include_top=False, weights="imagenet", input_tensor=inputs)

    # Encoder layers
    s1 = resnet50.layers[0].output         ## (512 x 512)
    s2 = resnet50.get_layer("conv1_relu").output        ## (256 x 256)
    s3 = resnet50.get_layer("conv2_block3_out").output  ## (128 x 128)
    s4 = resnet50.get_layer("conv3_block4_out").output  ## (64 x 64)
    b1 = resnet50.get_layer("conv4_block6_out").output
    # Create encoder model
    encoder_model = Model(inputs=inputs, outputs=[s1, s2, s3, s4,b1])

    # Freeze layers
    encoder_model.trainable = False

    return encoder_model

def build_resnet50_unet(input_shape=(256, 256, 3)):
    """ Encoder """
    in_channels= 3
    out_channels = 3
    kernel_size = 3
    stride = 1
    padding = 0
    bayar_conv = BayarConv2d(in_channels, out_channels, kernel_size, stride, padding)
    inputs = Input(input_shape)
    x=bayar_conv(inputs)
    encoder_model = build_encoder(input_shape)
    encoder_layers=encoder_model(inputs)
    encoder_layers_bayer=encoder_model(x)
    num_layers = len(encoder_layers) 
    #list to store the concatenated layers
    concatenated_layers_list = []
    for layer,layer_bayar in zip(encoder_layers, encoder_layers_bayer):
        concatenated_layers=concatenate([layer,layer_bayar], axis=-1)
        output = conv3x3(in_planes=concatenated_layers.shape[-1], out_planes=layer.shape[-1])(concatenated_layers)
        concatenated_layers_list.append(output)
    s1,s2,s3,s4,b1=concatenated_layers_list
    ppm_output = PyramidPoolingModule()(b1)
    """ Decoder """
    d1 = decoder_block(ppm_output, s4, 256)                     ## (64 x 64)
    d2 = decoder_block(d1, s3, 128)                     ## (128 x 128)
    d3 = decoder_block(d2, s2, 64)                     ## (256 x 256)
    d4 = decoder_block(d3, s1, 32)                      ## (512 x 512)

    """ Output """

    """ Localization """
    localization_output = Conv2D(1, 1, padding="same", activation="sigmoid",name="localization_output")(d4)

    """ Classification """
    avg_pooled = keras.layers.GlobalAveragePooling2D()(ppm_output)
    classification_output = keras.layers.Dense(
        1, activation="sigmoid", name="classification_output"
    )(avg_pooled)
    model = Model(inputs, outputs=[localization_output,classification_output], name="ResNet50_U-Net")
    return model
