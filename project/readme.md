# RAVIR - Retinal Vessel Segmentation 
This directory contains code of network architectures, run experiments and code for loading and processing of the images. 

<b>Disclaimer: The majority of the code was written using the PyCharm collaboration tool. Consequently, the contribution 
frequency might be skewed, even though both team members contributed equally to this project/labs.</b>


## How to Run the Code?
 #todo: add running instructions 

The main.py file loads the trained models and does sample predictions on the validation set.  


## RAVIR Dataset
The dataset was taken from the grand-challenge.org page for the segmentation challenge. (https://ravir.grand-challenge.org)

There are 23 training and 19 test retinal vessel images in the dataset. Each image has 768x768 pixels, where each pixel 
corresponds to 12.5 microns in the physical space. All images are grayscale images and all images contain arteries and
veins. The ground truth segmentation masks contain 3 classes. (0: Background, 128: Arteries, 256: Veins) 

The objective of the challenge is to correctly segment veins and arteries in the image. 

### Challenges of the Dataset
- Small dataset
- Varying image quality (contrast as well es sharpness)
- Very thin structures
- High similarity between veins and arteries
- Arteries and veins overlap
- Large images
- Long range prediction of the vessels

### Metrics
Since the challenge is ranked by using the mean DSC score, we used mean DSC (averaged over all foreground classes).
Furthermore, Jaccard is reported in the challenge, thus we used it as a metric as well. In addition to the metrics used 
in the challenge, we also report recall and precision.  

#### Sample Images
<img src="dataset/train/training_images/IR_Case_038.png" alt="Training image IR_Case_038" width="150"/>
<img src="dataset/train/training_masks/IR_Case_038.png" alt="Training image IR_Case_038" width="150"/>
<br>
<img src="dataset/train/training_images/IR_Case_055.png" alt="Training image IR_Case_038" width="150"/>
<img src="dataset/train/training_masks/IR_Case_055.png" alt="Training image IR_Case_038" width="150"/>

## Segmentation Pipeline
This segment describes the whole segmentation pipeline that was used int the experiments.
### Preprocessing
The intensity of the images was rescaled to 1th and 99th percentile. When loading the images, they were normalized 
between 0 and 1. Furthermore, random patched of size 256x256 were drawn for training the network to reduce memory 
demand and as a kind of augmentation. 

The masks were loaded and converted from grayscale to one-hot encoding. Furthermore, the corresponding patched were 
drawn for training of the network.

All images and masks were loaded in the GPU to increase training time and reduce CPU memory usage.

### Training
For all experiments, the Adam optimizer [[4]](#4) was used. Furthermore, for the hyperparameter optimizatiion, the 
training dataset was split into 19 train images and 4 validation images and trained the networks using the leave-1-out 
approach. For the final evaluation of our algorithms, we trained the networks on all available training data and uploaded
the predictions of the test set to the challenge website.

All used network architectures were modifications of the original U-Net <a id="1">[1]</a> paper.

### Postprocessing
The trained network was applied on all images in the dataset. All predictions were converted from one-hot-encoding to
grayscale segmentation maps that correspond to the challenge requirements using argmax.  

## CNN Architectures used in the challenge
### U-Net [[1]](#1)

<figure>
<img src="https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png" alt="Base ResUnet diagram" width="600"/>
<figcaption>Original U-Net diagram</figcaption>
</figure>

### Residual U-Net [[2]](#2)
Residual U-Net allows us to take advantage of U-Net in which we combine low level detail information with high level 
semantic information and Residual networks using identity mapping [[5]](#5) that facilitates training.
Residual unet

<figure>
<img src="https://ieeexplore.ieee.org/mediastore_new/IEEE/content/media/8859/8344090/8309343/liu2-2802944-large.gif" alt="Base ResUnet diagram" width="300"/>
<figcaption>Original ResUnet diagram</figcaption>
</figure>


## Experiments
### Setting Up The Baseline U-Net
First, we set up a baseline U-Net with slight modifications to the original U-Net paper.
(i.e. batch normalization, spatial dropout, padding: same) This model was used to compare the results against to and as a 
starting point for the following experiments. 

The hyperparameters were determined using a grid search. Following hyperparameters were included in the grid search:
```python
grd_srch_n_base = [16, 32, 64]
kernel_size = [(3, 3), (5, 5)]
learning_rate = [0.001, 0.0001]
alphas = np.array([1, 0.8, 0.6, 0.4])
```
All other hyperparameters were determined using experience from prior labs in the CM2003 course.

In the baseline U-Net, no augmentation was used. 

The objective of this experiment was to choose the set of hyperparameters that results in the best DSC. 

### Data Augmentation
In addition to the search of the best hyperparameters for our model, we implement the following data augmentation:
```python
rotation_range = 10,
width_shift_range = 0.2,
height_shift_range = 0.2,
zoom_range = [0.1, 1.4],
horizontal_flip = True,
fill_mode = 'reflect',
```

### Binary Classification
Since the precious experiments showed worse results than expected, the main hypothesis was that the vessels are
detected but classified poorly into arteries and veins. Thus, we performed this experiment, where we trained a U-Net 
using the result of the grid search as hyperparameters but using only binary segmentation maps. (i.e. arteries and veins
were merged into class "vessel")

### Training Using Weight Maps
In order to improve the detection of veins and arteries we try implementing weight maps which would add additional 
loss penalty to the most important features in images. Weight maps were created using <em>scikit-image</em> package by 
dilation of the provided original segmentation masks using a ```sitk.Ball``` kernel shape with a radius 2. 
Code for creating weight maps provided in <em>[preprocessing.py](code/preprocessing.py)</em>



|            Dilation weight map for IR_Case_034            |           Original mask for IR_Case_034           |
|:---------------------------------------------------------:|:-------------------------------------------------:|
| ![](dataset/train/training_masks_dilated/IR_Case_034.png) | ![](dataset/train/training_masks/IR_Case_034.png) |


### Setting Up The Residual U-Net
Original Residual U-Net has been modified by adding one more residual block to increase the depth of the network.
Due to a small dataset, the original ResUnet has been modified by adding spatial dropout.
Used Architecture can be found in <em>[ResUnet.py](code/ResUnet.py)</em>

## Results and Discussions
### Grid Search

| n_base | learning_rate | alpha | kernel_size | augmentation | epochs |   loss    | val_loss | dice_coef | val_dice_coef | precision | val_precision |   recall    | val_recall | jaccard  | val_jaccard |
|:------:|:-------------:|:-----:|:-----------:|:------------:|:------:|:---------:|:--------:|:---------:|:-------------:|:---------:|:-------------:|:-----------:|:----------:|:--------:|:-----------:|
|   64   |    0.0001     |  0.4  |   [3, 3]    |     None     |  700   | 0.131769  | 0.209682 | 0.805162  |   0.654791    | 0.953546  |   0.936545    |  0.953546   |  0.936545  | 0.911267 |  0.880820   |
|   64   |    0.0001     |  0.6  |   [3, 3]    |     None     |  700   | 0.197164  | 0.269131 | 0.739774  |   0.637601    | 0.942663  |   0.930478    |  0.942663   |  0.930478  | 0.891680 |  0.870212   |
|   32   |    0.0001     |  0.6  |   [3, 3]    |     None     |  700   | 0.229841  | 0.270891 | 0.692221  |   0.635036    | 0.937279  |   0.929765    |  0.937279   |  0.929765  | 0.882102 |  0.869073   |
|   32   |    0.0001     |  0.8  |   [3, 3]    |     None     |  700   | 0.317092  | 0.335769 | 0.651637  |   0.620721    | 0.930788  |   0.922410    |  0.930788   |  0.922410  | 0.870864 |  0.856485   |
|   32   |    0.0001     |  0.4  |   [3, 3]    |     None     |  700   | 0.162371  | 0.236693 | 0.747658  |   0.616998    | 0.944697  |   0.922116    | 0.944697    |  0.922116  | 0.895331 |  0.855859   |


### Baseline Model


| Dataset |   Mean Dice   |  Artery Dice  |   Vein Dice   | Mean Jaccard  | Artery Jaccard |
|:-------:|:-------------:|:-------------:|:-------------:|:-------------:|:--------------:|
|  Test   | 0.4208±0.0586 | 0.3599±0.0535 | 0.4818±0.0646 | 0.2702±0.0483 | 0.2207±0.0402  |


### Binary classification
|                  Loss                  |               Dice Score               |                  Precision                  |                  Recall                  |
|:--------------------------------------:|:--------------------------------------:|:-------------------------------------------:|:----------------------------------------:|
| ![](predictions/binary_model/loss.png) | ![](predictions/binary_model/dice.png) | ![](predictions/binary_model/precision.png) | ![](predictions/binary_model/recall.png) |

It can be seen in the learning curves above, that the training metrics as well as the validation metrics all increase
when using a binary classification. The loss and all metrics seem to be converged. Furthermore, it should be noted, 
that there is a difference between the training and validation loss and metrics. Since the validation loss is higher and
the validation metrics are lower than the respective curves for the training data, it can be concluded that the model is 
slightly overfitting. 

#### Example of Bad Segmentation
The image IR_Case_058.png was segmented poorly. In the following image, the difference between the ground truth and 
prediction can be seen. The <b>black pixel denote pixels that have been segmented correctly, the white pixel denote false 
positive segmentations</b> and the <b>gray pixels denote false negative pixels</b>. It can be seen that the boundaries are not 
segmented perfectly and also that some vessels were not detected. The Dice sore for the following segmentation is 0.74.

<img src="predictions/binary_model/IR_Case_058_binary_difference.png" alt="Training image IR_Case_038" width="400"/>

#### Discussion
It can be seen that the metrics for the binary classifications are substantially higher than the metrics for the 
multiclasss-classification. It seems to be the case, that the baseline U-Net is able to segment the vessels quite 
robustly and that the challenge in this dataset is not necessarily the segmentation of the vessels but rather the correct
classification of veins and arteries. This seems to require long range prediction, as some vessels change their class 
along the vessel in the prediction of our multiclass models. 

Furthermore, it should be noted that the binary model achieves a Dice score of around 0.8 on the validiation data.
Even though, this dice score is reasonably high, the network fails to properly segment the thin parts of the vessels. This
might be caused by under-representation of thin vessels in the Dice score. (i.e. there are more thick vessels that are 
easier to segment and consequently weigh more in the dice score) 



### Baseline ResUnet Model
First we train our baseline ResUnet model using a train/validation split described in the [Training](#training) section to see how it performs.

|                                  Loss                                  |                          Dice Score                          |                               Precision                                |                               Recall                                |                            Jaccard Score                             |
|:----------------------------------------------------------------------:|:------------------------------------------------------------:|:----------------------------------------------------------------------:|:-------------------------------------------------------------------:|:--------------------------------------------------------------------:|
| ![](predictions/ResUnet_baseline_with_val/curves/dice_coef_metric.png) | ![](predictions/ResUnet_baseline_with_val/curves/losses.png) | ![](predictions/ResUnet_baseline_with_val/curves/precision_metric.png) | ![](predictions/ResUnet_baseline_with_val/curves/recall_metric.png) | ![](predictions/ResUnet_baseline_with_val/curves/jaccard_metric.png) |

In the learning curves above we can see that the model is overfitting and that the validation curve reaches a plateau 
after approximately 250 epochs and does not significantly increase during further training however the train metrics keep improving.

Metrics presented below were plotted for the training dataset consisting of all train images without the validation set.

|                             Loss                              |                     Dice Score                      |                           Precision                           |                           Recall                           |                        Jaccard Score                        |
|:-------------------------------------------------------------:|:---------------------------------------------------:|:-------------------------------------------------------------:|:----------------------------------------------------------:|:-----------------------------------------------------------:|
| ![](predictions/ResUnet_baseline/curves/dice_coef_metric.png) | ![](predictions/ResUnet_baseline/curves/losses.png) | ![](predictions/ResUnet_baseline/curves/precision_metric.png) | ![](predictions/ResUnet_baseline/curves/recall_metric.png) | ![](predictions/ResUnet_baseline/curves/jaccard_metric.png) |

Predictions from the model trained on all images were uploaded to the RAVIR challenge for evaluation and resulted in the following results:

| Dataset |   Mean Dice    |  Artery Dice  |   Vein Dice   | Mean Jaccard  | Artery Jaccard |
|:-------:|:--------------:|:-------------:|:-------------:|:-------------:|:--------------:|
|  Test   | 	0.4290±0.0583 | 0.3673±0.0531 | 0.4906±0.0645 | 0.2768±0.0478 | 0.2262±0.0400  |

We notice that the mean Dice and Jaccard Scores are lower when computed on the challenge are lower than the ones computed
by us which means that metrics we use are not representative of the actual scores


## Suggested further improvements/ideas
- ResUnet-a is based on the Residual U-Net which uses diluted convolutions to improve the receptive field of the network [[3]](#3)
- Using weight maps that have a higher weight for thinner structures to emphasize those regions in the training.

## References
<a id="1">[1]</a>
O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional Networks for Biomedical Image Segmentation.” arXiv, May 18, 2015. doi: 10.48550/arXiv.1505.04597.

<a id="2">[2]</a>
Z. Zhang, Q. Liu, and Y. Wang, “Road Extraction by Deep Residual U-Net,” IEEE Geosci. Remote Sensing Lett., vol. 15, no. 5, pp. 749–753, May 2018, doi: 10.1109/LGRS.2018.2802944.

<a id="3">[3]</a> 
F. I. Diakogiannis, F. Waldner, P. Caccetta, and C. Wu, “ResUNet-a: a deep learning framework for semantic segmentation of remotely sensed data”, ISPRS Journal of Photogrammetry and Remote Sensing, vol. 162, pp. 94–114, Apr. 2020, doi: 10.1016/j.isprsjprs.2020.01.013.

<a id="4">[4]</a> 
D. P. Kingma and J. Ba, “Adam: A Method for Stochastic Optimization,” in 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings, 2015. [Online]. Available: http://arxiv.org/abs/1412.6980

<a id="5">[5]</a> 
K. He, X. Zhang, S. Ren, and J. Sun, “Identity Mappings in Deep Residual Networks.” arXiv, Jul. 25, 2016. doi: 10.48550/arXiv.1603.05027.



