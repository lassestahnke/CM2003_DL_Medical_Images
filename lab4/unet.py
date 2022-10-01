# function to define unet architecture
from tensorflow.keras.layers import Conv2D, MaxPool2D, Activation, BatchNormalization, Conv2DTranspose, concatenate, SpatialDropout2D
from tensorflow.keras import Input
from tensorflow.keras.models import Model

def conv_block(input, filters, use_batch_norm):
    conv_1 = Conv2D(filters, kernel_size=(3,3), padding='same')(input)
    if use_batch_norm:
        conv_1 = BatchNormalization()(conv_1)
    relu_1 = Activation('relu')(conv_1)

    conv_2 = Conv2D(filters, kernel_size=(3,3), padding='same')(relu_1)
    if use_batch_norm:
        conv_2 = BatchNormalization()(conv_2)
    relu_2 = Activation('relu')(conv_2)

    return relu_2

def get_unet(input_shape, n_classes, batch_size, n_base, dropout_rate=0, use_batch_norm=False):
    # define input
    input = Input(shape=input_shape)
    # define encoder
    # level 1
    level_1_enc = conv_block(input=input, filters=n_base, use_batch_norm=use_batch_norm)
    level_1_max_pool = MaxPool2D(pool_size=(2,2))(level_1_enc)
    level_1_max_pool = SpatialDropout2D(dropout_rate)(level_1_max_pool)

    # level 2
    level_2_enc = conv_block(input=level_1_max_pool, filters=2*n_base, use_batch_norm=use_batch_norm)
    level_2_max_pool = MaxPool2D(pool_size=(2, 2))(level_2_enc)
    level_2_max_pool = SpatialDropout2D(dropout_rate)(level_2_max_pool)

    # level 3
    level_3_enc = conv_block(level_2_max_pool, filters=4*n_base, use_batch_norm=use_batch_norm)
    level_3_max_pool = MaxPool2D(pool_size=(2,2))(level_3_enc)
    level_3_max_pool = SpatialDropout2D(dropout_rate)(level_3_max_pool)

    # level 4
    level_4_enc = conv_block(level_3_max_pool, filters=8*n_base, use_batch_norm=use_batch_norm)
    level_4_max_pool = MaxPool2D(pool_size=(2,2))(level_4_enc)
    level_4_max_pool = SpatialDropout2D(dropout_rate)(level_4_max_pool)
    # Bottleneck / level 5
    bottleneck = conv_block(level_4_max_pool, filters=16*n_base, use_batch_norm=use_batch_norm)

    # define decoder
    # level 4
    bottleneck_up_conv = Conv2DTranspose(filters=8*n_base, kernel_size=(3,3), strides=(2,2), padding='same')(bottleneck)
    level_4_concat = concatenate([bottleneck_up_conv, level_4_enc])
    level_4_concat = SpatialDropout2D(dropout_rate)(level_4_concat)
    level_4_dec = conv_block(level_4_concat, filters=8*n_base, use_batch_norm=use_batch_norm)

    # level 3
    level_4_up_conv = Conv2DTranspose(filters=4*n_base, kernel_size=(3,3), strides=(2,2), padding='same')(level_4_dec)
    level_3_concat = concatenate([level_4_up_conv, level_3_enc])
    level_3_concat = SpatialDropout2D(dropout_rate)(level_3_concat)
    level_3_dec = conv_block(level_3_concat, filters=4*n_base, use_batch_norm=use_batch_norm)

    # level 2
    level_3_up_conv = Conv2DTranspose(filters=2*n_base, kernel_size=(3,3), strides=(2,2), padding='same')(level_3_dec)
    level_2_concat = concatenate([level_3_up_conv, level_2_enc])
    level_2_concat = SpatialDropout2D(dropout_rate)(level_2_concat)
    level_2_dec = conv_block(level_2_concat, filters=2*n_base, use_batch_norm=use_batch_norm)

    # level 1
    level_2_up_conv = Conv2DTranspose(filters=n_base, kernel_size=(3,3), strides=(2,2), padding='same')(level_2_dec)
    level_1_concat = concatenate([level_2_up_conv, level_1_enc])
    level_1_concat = SpatialDropout2D(dropout_rate)(level_1_concat)
    level_1_dec = conv_block(level_1_concat, filters=n_base, use_batch_norm=use_batch_norm)

    # output layer
    if n_classes == 1:
        output = Conv2D(n_classes, kernel_size=(1,1), padding='same', activation='sigmoid')(level_1_dec)
    else:
        output = Conv2D(n_classes, kernel_size=(1, 1), padding='same', activation='softmax')(level_1_dec)

    model = Model(inputs=input, outputs=output)
    return model












