from metrics import dice_coef

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)