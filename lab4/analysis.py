# define function for visualization of learning curves:
import matplotlib.pyplot as plt
import numpy as np

def learning_curves(history, loss_key, validation_loss_key, metric_keys, validation_metric_keys,
                    loss_range=(0,1), metric_range=(0,1)):
    """
        Function to plot learning curves.

    :param history: [str] keras model history
    :param loss_key: [str] name of loss key in hist_name
    :param validation_loss_key: [str] name of validation loss key in hist_name
    :param metric_keys: [list]: used metrics in model
    :param validation_metric_keys: [list]: used validation metrics in model
    :param loss_range: [tuple] range for loss plot on y-axis
    :param metric_range: [tuple] range for metric plot on y-axis
    :return:
    """
    plt.figure(figsize=(4, 4))
    plt.title("Learning curve")
    plt.plot(history.history[loss_key], label=loss_key) #loss is training loss
    plt.plot(history.history[validation_loss_key], label=validation_loss_key) #val_loss is validation loss
    plt.plot(np.argmin(history.history[validation_loss_key]),
             np.min(history.history[validation_loss_key]), marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.ylim(loss_range)

    plt.legend()
    plt.show()
    print(history.history.keys())

    for i in range(len(metric_keys)):
        print(metric_keys)
        print(metric_keys[i], history.history[metric_keys[i]][-1])
        print(validation_metric_keys[i], history.history[validation_metric_keys[i]][-1])


    # plotting metric curves
    for met in metric_keys:
        print(met)
        plt.figure(figsize=(4, 4))
        plt.title("Learning curve")
        plt.plot(history.history[met], label=met) # training accuracy
        plt.plot(history.history[met], label=met) # validation accuracy
        plt.xlabel("Epochs")
        plt.ylabel(met)
        plt.ylim(metric_range)
        plt.legend()
        plt.show()
    return