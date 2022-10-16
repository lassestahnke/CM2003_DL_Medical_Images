# function to define unet architecture
from tensorflow.keras.layers import Conv2D, MaxPool2D, Activation, BatchNormalization, Conv2DTranspose, concatenate, SpatialDropout2D, Reshape, ConvLSTM2D
from tensorflow.keras import Input
from tensorflow.keras.models import Model
import numpy as np

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

def get_unet(input_shape, n_classes, n_base, dropout_rate=0, use_batch_norm=False, boundary_path = None):
    # define input
    input = Input(shape=input_shape)

    if boundary_path is not None:
        loss_weights = Input(shape=(input_shape[0], input_shape[1], 1), name="loss_weights")

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
    bottleneck_up_conv = Conv2DTranspose(filters=8*n_base,
                                         kernel_size=(3,3),
                                         strides=(2,2),
                                         padding='same')(bottleneck)

    x1 = Reshape(target_shape=(1, np.int32(input_shape[0] / 8),
                               np.int32(input_shape[1] / 8),
                               n_base * 8))(level_4_enc)

    x2 = Reshape(target_shape=(1, np.int32(input_shape[0] / 8),
                               np.int32(input_shape[1] / 8),
                               n_base * 8))(bottleneck_up_conv)
    level_4_concat = concatenate([x1, x2], axis=1)
    level_4_lstm = ConvLSTM2D(n_base*4, (3, 3), padding='same', return_sequences=False, go_backwards=True)(level_4_concat)
    level_4_lstm = SpatialDropout2D(dropout_rate)(level_4_lstm)
    level_4_dec = conv_block(level_4_lstm, filters=8 * n_base, use_batch_norm=use_batch_norm)

    # level 3
    level_4_up_conv = Conv2DTranspose(filters=4*n_base,
                                         kernel_size=(3,3),
                                         strides=(2,2),
                                         padding='same')(level_4_dec)

    x1 = Reshape(target_shape=(1, np.int32(input_shape[0] / 4),
                               np.int32(input_shape[1] / 4),
                               n_base * 4))(level_3_enc)

    x2 = Reshape(target_shape=(1, np.int32(input_shape[0] / 4),
                               np.int32(input_shape[1] / 4),
                               n_base * 4))(level_4_up_conv)
    level_3_concat = concatenate([x1, x2], axis=1)
    level_3_lstm = ConvLSTM2D(n_base*4, (3, 3), padding='same', return_sequences=False, go_backwards=True)(level_3_concat)
    level_3_lstm = SpatialDropout2D(dropout_rate)(level_3_lstm)
    level_3_dec = conv_block(level_3_lstm, filters=4 * n_base, use_batch_norm=use_batch_norm)

    # level 2
    level_3_up_conv = Conv2DTranspose(filters=2*n_base,
                                         kernel_size=(3,3),
                                         strides=(2,2),
                                         padding='same')(level_3_dec)

    x1 = Reshape(target_shape=(1, np.int32(input_shape[0] / 2),
                               np.int32(input_shape[1] / 2),
                               n_base * 2))(level_2_enc)

    x2 = Reshape(target_shape=(1, np.int32(input_shape[0] / 2),
                               np.int32(input_shape[1] / 2),
                               n_base * 2))(level_3_up_conv)
    level_2_concat = concatenate([x1, x2], axis=1)
    level_2_lstm = ConvLSTM2D(n_base*2, (3, 3), padding='same', return_sequences=False, go_backwards=True)(level_2_concat)
    level_2_lstm = SpatialDropout2D(dropout_rate)(level_2_lstm)
    level_2_dec = conv_block(level_2_lstm, filters=2 * n_base, use_batch_norm=use_batch_norm)


    # level 1
    level_2_up_conv = Conv2DTranspose(filters=n_base,
                                         kernel_size=(3,3),
                                         strides=(2,2),
                                         padding='same')(level_2_dec)

    x1 = Reshape(target_shape=(1, np.int32(input_shape[0]),
                               np.int32(input_shape[1]),
                               n_base))(level_1_enc)

    x2 = Reshape(target_shape=(1, np.int32(input_shape[0]),
                               np.int32(input_shape[1]),
                               n_base))(level_2_up_conv)
    level_1_concat = concatenate([x1, x2], axis=1)
    level_1_lstm = ConvLSTM2D(n_base, (3, 3), padding='same', return_sequences=False, go_backwards=True)(level_1_concat)
    level_1_lstm = SpatialDropout2D(dropout_rate)(level_1_lstm)
    level_1_dec = conv_block(level_1_lstm, filters=n_base, use_batch_norm=use_batch_norm)

    # output layer
    if n_classes == 1:
        output = Conv2D(n_classes, kernel_size=(1,1), padding='same', activation='sigmoid')(level_1_dec)
    else:
        output = Conv2D(n_classes, kernel_size=(1, 1), padding='same', activation='softmax')(level_1_dec)

    if boundary_path is not None:
        model = Model(inputs=[input, loss_weights] , outputs=output)
        return model, loss_weights
    else:
        model = Model(inputs=input , outputs=output)
        return model












