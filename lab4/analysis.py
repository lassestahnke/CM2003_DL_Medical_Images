# define function for visualization of learning curves:
import matplotlib.pyplot as plt
import numpy as np

def learning_curves(hist_name, loss_key, validation_loss_key, metric_key, validation_metric_key,
                    loss_range=(0,1), metric_range=(0,1)):
    plt.figure(figsize=(4, 4))
    plt.title("Learning curve")
    plt.plot(hist_name.history[loss_key], label=loss_key) #loss is training loss
    plt.plot(hist_name.history[validation_loss_key], label=validation_loss_key) #val_loss is validation loss
    plt.plot(np.argmin(hist_name.history[validation_loss_key]),
    np.min(hist_name.history[validation_loss_key]), marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.ylim(loss_range)

    plt.legend()
    print(hist_name.history.keys())
    print("Training metric", hist_name.history[metric_key][-1])
    print("Validation metric", hist_name.history[validation_metric_key][-1])
    plt.show()

    # plotting accuracy curves
    plt.figure(figsize=(4, 4))
    plt.title("Learning curve")
    plt.plot(hist_name.history[metric_key], label=metric_key) # training accuracy
    plt.plot(hist_name.history[validation_metric_key], label=validation_metric_key) # validation accuracy
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.ylim(metric_range)
    plt.show()
    print('a')
    return