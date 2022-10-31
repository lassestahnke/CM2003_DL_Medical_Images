from metrics import dice_coef
from tensorflow.keras.losses import BinaryCrossentropy

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def combined_loss(y_true, y_pred):
    bce = BinaryCrossentropy()
    return 0.7 * dice_loss(y_true,y_pred) + 0.3 * bce(y_true, y_pred)