from metrics import dice_coef
from tensorflow.keras.losses import BinaryCrossentropy
import tensorflow.keras.backend as K

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def combined_loss(alpha):
    def loss(y_true, y_pred):
        bce = BinaryCrossentropy()
        return alpha * dice_loss(y_true, y_pred) + (1 - alpha) * bce(y_true, y_pred)

    return loss

def weighted_loss(weight_map, weight_strength):
    def weighted_dice_loss(y_true, y_pred):
        y_true_f = K.flatten(y_true[...,1:])
        y_pred_f = K.flatten(y_pred[...,1:])
        print("weight map shape", weight_map.shape)
        weight_f = K.flatten(weight_map)
        print("weight_f shape", weight_f.shape)
        weight_f = K.concatenate([weight_f, weight_f])
        print("weight_f concat shape", weight_f.shape)
        weight_f = weight_f * weight_strength
        weight_f = weight_f + 1
        weighted_intersection = K.sum(weight_f * (y_true_f * y_pred_f))
        return -(2. * weighted_intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())
    return weighted_dice_loss

