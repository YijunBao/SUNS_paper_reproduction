import tensorflow as tf
from tensorflow.python.keras import losses
# tf.config.gpu.set_per_process_memory_fraction(0.5)

def dice_coeff(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f ** 2) + tf.reduce_sum(y_pred_f ** 2) + smooth)
    # score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    loss = 20*losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    # loss = dice_loss(y_true, y_pred)
    return loss

def get_shallow_unet(): #(size)
    # inputs = tf.keras.layers.Input((size, size, 1))
    inputs = tf.keras.layers.Input((None, None, 1))
    # s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

    c1 = tf.keras.layers.Conv2D(4, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(inputs)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    #c1 = tf.keras.layers.Conv2D(4, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
    #                            padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.SeparableConv2D(8, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(p1)
    # c2 = tf.keras.layers.Conv2D(8, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
    #                             padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    #c2 = tf.keras.layers.Conv2D(8, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
    #                            padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = tf.keras.layers.SeparableConv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(p2)
    # c3 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
    #                             padding='same')(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    #c3 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
    #                            padding='same')(c3)
    #p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

    #c4 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
    #                            padding='same')(p3)
    #c4 = tf.keras.layers.Dropout(0.2)(c4)
    #c4 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
    #                            padding='same')(c4)
    #p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

    ##c5 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
    ##                            padding='same')(p4)
    ##c5 = tf.keras.layers.Dropout(0.3)(c5)
    #c5 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
    #                            padding='same')(c5)

    ##u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    #u6 = tf.keras.layers.concatenate([u6, c4])

    #c6 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
    #                            padding='same')(c4)
    #c6 = tf.keras.layers.Dropout(0.2)(c6)
    #c6 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
    #                            padding='same')(c6)

    #u7 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c4)
    #u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(c3)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    #c7 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
    #                            padding='same')(c7)

    u8 = tf.keras.layers.Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same')(c7)
    # u8 = tf.keras.layers.concatenate([u8, c2], axis=3)
    c8 = tf.keras.layers.Conv2D(8, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(u8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    #c8 = tf.keras.layers.Conv2D(8, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
    #                            padding='same')(c8)

    u9 = tf.keras.layers.Conv2DTranspose(4, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    c9 = tf.keras.layers.Conv2D(4, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(u9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    #c9 = tf.keras.layers.Conv2D(4, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
    #                            padding='same')(c9)

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    # model.compile(optimizer='adam', loss=dice_loss, metrics=[dice_loss])
    model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[dice_loss])
    # model.summary()
    return model
