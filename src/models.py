import tensorflow as tf
from tensorflow.keras import layers

def build_generator():
    fabric_input = layers.Input(shape=(256, 256, 3))

    # Fabric Style Encoder
    x = layers.Conv2D(64, 3, strides=2, padding='same')(fabric_input)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, 3, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)
    fabric_style = layers.Dense(512, activation='relu')(x)

    # Broadcast style to spatial tensor
    style_broadcast = layers.Dense(8 * 8 * 512, activation='relu')(fabric_style)
    style_broadcast = layers.Reshape((8, 8, 512))(style_broadcast)

    # Decoder (upsample only for now, assumes encoded input)
    up1 = upsample(512, 4, True)(style_broadcast)
    up2 = upsample(512, 4, True)(up1)
    up3 = upsample(256, 4)(up2)
    up4 = upsample(128, 4)(up3)
    up5 = upsample(64, 4)(up4)

    last = layers.Conv2DTranspose(3, 4, strides=2, padding='same', activation='tanh')(up5)

    return tf.keras.Model(inputs=fabric_input, outputs=last)

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                      kernel_initializer=initializer, use_bias=False))
    result.add(layers.BatchNormalization())
    if apply_dropout:
        result.add(layers.Dropout(0.5))
    result.add(layers.ReLU())
    return result

def build_discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)
    inp = layers.Input(shape=[256, 256, 3])
    tar = layers.Input(shape=[256, 256, 3])

    x = layers.concatenate([inp, tar])
    down1 = layers.Conv2D(64, 4, strides=2, padding='same', kernel_initializer=initializer)(x)
    down1 = layers.LeakyReLU()(down1)

    down2 = layers.Conv2D(128, 4, strides=2, padding='same', kernel_initializer=initializer)(down1)
    down2 = layers.BatchNormalization()(down2)
    down2 = layers.LeakyReLU()(down2)

    down3 = layers.Conv2D(256, 4, strides=2, padding='same', kernel_initializer=initializer)(down2)
    down3 = layers.BatchNormalization()(down3)
    down3 = layers.LeakyReLU()(down3)

    zero_pad1 = layers.ZeroPadding2D()(down3)
    conv = layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer)(zero_pad1)
    batchnorm = layers.BatchNormalization()(conv)
    leaky_relu = layers.LeakyReLU()(batchnorm)
    zero_pad2 = layers.ZeroPadding2D()(leaky_relu)
    last = layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)
