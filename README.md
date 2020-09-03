# XAI_pcam
Interpretable Explanations of Lymph Node Metastases using Extremal Perturbations and GRAD-CAM.

![Saliency Heatmaps.](https://github.com/ThomasAllcock/XAI_pcam/blob/master/positive_comp.png)
*Extremal perturbation and GRAD-CAM saliency heatmaps for each model. Original image with ground-truth segmentation overlaid is on the far left. All images have a positive label and are correctly classified by each model.*

## Abstract
Deep convolutional neural networks (CNNs) are increasingly being applied to histopathology. However, in order for deep CNNs to be implemented into clinical use, methods to explain a decision must be explored. Two contributions are made in this paper. The first is to explore the potential of two attribution methods to explain the output of a model. Attribution works by finding the part of a model's input most responsible for the output. The attribution methods used in the experiments are extremal perturbation and gradient-weighted class activation mapping (GRAD-CAM). The dataset used is Patch Camelyon (P-CAM). The second contribution is to show that relevant image information can be extracted without additional annotated segmentation data. Both of these are important for increasing the interpretability and explainable power of deep networks, ultimately to create models more suitable for clinical use.  

## Dataset
P-CAM dataset was used for training the CNNs. It can be downloaded here - https://github.com/basveeling/pcam.
The .h5 files contain the patches for training, validating and testing each model. The meta.csv files are required for matching each patch up with the ground-truth segmentations
which show the locations of metastases in each patch.

P-CAM contains 96x96px patches from the CAMELYON16 challenge dataset which is available here - https://camelyon17.grand-challenge.org/Data/

The ground-truth segmentations for the test set are provided in the testset_mask files. These are whole slide image (WSI) segmentations. Using the meta data csv files for the test set, the patchs in P-CAM can be matched up with their ground-truth segmentation.

## Models
Weights for three models trained on the P-CAM dataset are available here -  https://drive.google.com/drive/folders/1axdMD12LLHjeVTpavs65TUYOT7p3W2a1?usp=sharing. These weights are for an 11-layer VGG, a 34-layer ResNet and a 22-layer GoogLeNet.
The checkpoint folder provides the weights directly after training. The best_model folder provides the weights which achieved the lowest loss. 
The .py files contain a function (load_ckp) for loading the weights and the code for each model's architecture is there too.

## TorchRay
TorchRay is a python package that is required for the explainable AI used here. This can be found here - https://github.com/facebookresearch/TorchRay
