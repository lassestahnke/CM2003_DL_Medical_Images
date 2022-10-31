from metrics import dice_coef
from tensorflow.keras.losses import BinaryCrossentropy

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def combined_loss(alpha):
    def loss(y_true, y_pred):
        bce = BinaryCrossentropy()
        return alpha * dice_loss(y_true,y_pred) + (1-alpha) * bce(y_true, y_pred)
    return loss