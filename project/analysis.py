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
          sort_by [str]: Name of column in DataFrame to sort
    return:
        pandas.dataframe of all experiments
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

def segment_from_directory(pred_dir, model, base_dir, dir):
    """ Function to segment images from base_dir/dir directory. New directory of the same name is created within the
        pred_dir directory.

        args:
            pred_dir [str]: Directory to save predictions
            model [tensorflow.Model] model used for predictions
            base_dir [str]: directory of folder that contains images to segment
            dir: [str]: directory that contains images to segment
    """

    # create pred_dir/dir if not available
    if not os.path.exists(os.path.join(pred_dir, dir)):
        os.makedirs(os.path.join(pred_dir, dir))

    for file in os.listdir(os.path.join(base_dir, dir)):
        if not file.endswith(".png"):
            continue
        # load image img
        img = plt.imread(os.path.join(base_dir, dir, file), format="png")
        # use model to predict segmentation
        pred = model.predict(img[None, :,:, None])
        #pred[pred > 0.25] = 1
        #pred[pred <= 0.25] = 0
        pred_new = np.argmax(pred[0, :, :, :], axis=2)
        pred_new[pred_new == 1] = 128
        pred_new[pred_new == 2] = 255
        print(pred[0, :, :, 0].max())
        print(pred[0, :, :, 1].max())
        print(pred[0, :, :, 2].max())



        #pred_new = np.zeros((pred.shape[1], pred.shape[2]))
        #pred_new[pred[0, :, :, 0] > pred[0, :, :, 1]] = 128
        #pred_new[pred[0, :, :, 0] < pred[0, :, :, 1]] = 255

        # modify pred to fulfil challenge requirements
        #prediction = np.zeros((pred.shape[0], pred.shape[0]))
        #prediction = pred[0, :, :, 0] * 128 + pred[0, :, :, 1] * 255
        print(np.unique(pred_new))
        # write image to base_dir,dir with same name
        plt.imsave(os.path.join(pred_dir, dir, file), pred_new, cmap="gray")
        print("saving: ", os.path.join(pred_dir, dir, file))
        plt.imshow(pred_new)
        plt.show()


