# define function for visualization of learning curves:
import json

import matplotlib.pyplot as plt
import numpy as np
import os

import pandas as pd


def learning_curves(history, loss_key, validation_loss_key, metric_keys, validation_metric_keys,
                    loss_range=(0, 1), metric_range=(0, 1), save_path=None):
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
    plt.plot(history.history[loss_key], label=loss_key)  # loss is training loss
    plt.plot(history.history[validation_loss_key], label=validation_loss_key)  # val_loss is validation loss
    plt.plot(np.argmin(history.history[validation_loss_key]),
             np.min(history.history[validation_loss_key]), marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.ylim(loss_range)

    plt.legend()

    if save_path is not None:
        plt.savefig(os.path.join(save_path,"losses.png"))

    plt.show()

    print(history.history.keys())

    for i in range(len(metric_keys)):
        print(metric_keys[i])
        print(metric_keys[i], history.history[metric_keys[i]][-1])
        print(validation_metric_keys[i], history.history[validation_metric_keys[i]][-1])

    # plotting metric curves
    for met in range(len(metric_keys)):
        print(metric_keys[met])
        plt.figure(figsize=(4, 4))
        plt.title("Learning curve")
        plt.plot(history.history[metric_keys[met]], label=metric_keys[met])  # training accuracy
        plt.plot(history.history[validation_metric_keys[met]], label=validation_metric_keys[met])  # validation accuracy
        plt.xlabel("Epochs")
        plt.ylabel(metric_keys[met])
        plt.ylim(metric_range)
        plt.legend()

        if save_path is not None:
            plt.savefig(os.path.join(save_path, "{}_metric.png".format(metric_keys[met])))

        plt.show()

    return


def sample_segmentation(data_loader, n_samples=1, model=None):
    for i in range(n_samples):
        img, msk = next(data_loader)

        # plot original label map
        plt.imshow(img[0, :, :, :], cmap="gray")
        plt.imshow(msk[0, :, :, :], cmap="jet", alpha=0.5)
        plt.title("Ground Truth")
        plt.show()

        # plot model prediction:
        if model is not None:
            plt.imshow(img[0, :, :, :], cmap="gray")
            pred = model.predict(img)
            plt.imshow(pred[0, :, :, :], cmap="jet", alpha=0.5)
            plt.title("Prediction")
            plt.show()

    return


def read_grid_search_json(path, sort_by="val_dice_coef"):
    """
    File to read .json files in folder, convert them to dict and wrap them into one pandas Dataframe
    args: path [str]: path to directory
    """
    data_frame = pd.DataFrame()
    for file in os.listdir(path):
        if not file.endswith(".json"):
            continue

        with open(os.path.join(path, file), "r") as file_read:
            print(file_read.name)
            dic = json.load(file_read)
            data_frame = data_frame.append(dic, ignore_index=True)

    return data_frame.sort_values(sort_by, ascending=False)

