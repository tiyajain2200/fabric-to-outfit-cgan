import tensorflow as tf
from tensorflow.keras import layers

# ------------------ Generator (U-Net) ------------------

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(layers.BatchNormalization())
    result.add(layers.LeakyReLU())
    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(layers.Conv2DTranspose(filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))
    result.add(layers.BatchNormalization())
    if apply_dropout:
        result.add(layers.Dropout(0.5))
    result.add(layers.ReLU())
    return result

def build_generator():
    inputs = layers.Input(shape=[256, 256, 3])

    # Encoder (downsampling)
    down1 = downsample(64, 4, apply_batchnorm=False)(inputs)  # (64x64)
    down2 = downsample(128, 4)(down1)                          # (32x32)
    down3 = downsample(256, 4)(down2)                          # (16x16)
    down4 = downsample(512, 4)(down3)                          # (8x8)
    down5 = downsample(512, 4)(down4)                          # (4x4)
    down6 = downsample(512, 4)(down5)                          # (2x2)
    down7 = downsample(512, 4)(down6)                          # (1x1)

    # Decoder (upsampling)
    up1 = upsample(512, 4, apply_dropout=True)(down7)
    up1 = layers.Concatenate()([up1, down6])

    up2 = upsample(512, 4, apply_dropout=True)(up1)
    up2 = layers.Concatenate()([up2, down5])

    up3 = upsample(512, 4, apply_dropout=True)(up2)
    up3 = layers.Concatenate()([up3, down4])

    up4 = upsample(256, 4)(up3)
    up4 = layers.Concatenate()([up4, down3])

    up5 = upsample(128, 4)(up4)
    up5 = layers.Concatenate()([up5, down2])

    up6 = upsample(64, 4)(up5)
    up6 = layers.Concatenate()([up6, down1])

    initializer = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(3, 4,
                                  strides=2,
                                  padding='same',
                                  kernel_initializer=initializer,
                                  activation='tanh')  # (128x128x3)

    x = last(up6)

    return tf.keras.Model(inputs=inputs, outputs=x)

# ------------------ Discriminator (PatchGAN) ------------------

def build_discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)
    inp = layers.Input(shape=[256, 256, 3], name='input_image')
    tar = layers.Input(shape=[256, 256, 3], name='target_image')

    x = layers.concatenate([inp, tar])  # (128x128x6)

    down1 = layers.Conv2D(64, 4, strides=2, padding='same',
                          kernel_initializer=initializer)(x)
    down1 = layers.LeakyReLU()(down1)

    down2 = layers.Conv2D(128, 4, strides=2, padding='same',
                          kernel_initializer=initializer)(down1)
    down2 = layers.BatchNormalization()(down2)
    down2 = layers.LeakyReLU()(down2)

    down3 = layers.Conv2D(256, 4, strides=2, padding='same',
                          kernel_initializer=initializer)(down2)
    down3 = layers.BatchNormalization()(down3)
    down3 = layers.LeakyReLU()(down3)

    zero_pad1 = layers.ZeroPadding2D()(down3)
    conv = layers.Conv2D(512, 4, strides=1,
                         kernel_initializer=initializer)(zero_pad1)
    batchnorm = layers.BatchNormalization()(conv)
    leaky_relu = layers.LeakyReLU()(batchnorm)
    zero_pad2 = layers.ZeroPadding2D()(leaky_relu)

    last = layers.Conv2D(1, 4, strides=1,
                         kernel_initializer=initializer)(zero_pad2)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)
