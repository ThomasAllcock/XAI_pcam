# XAI_pcam
Interpretable Explanations of Lymph Node Metastases using Extremal Perturbations and GRAD-CAM.

![Saliency Heatmaps.](https://github.com/ThomasAllcock/XAI_pcam/blob/master/positive_comp.png)
*Extremal perturbation and GRAD-CAM saliency heatmaps for each model. Original image with ground-truth segmentation overlaid is on the far left. All images have a positive label and are correctly classified by each model.*

## Abstract
Deep convolutional neural networks (CNNs) are increasingly being applied to histopathology. However, in order for deep CNNs to be implemented into clinical use, methods to explain a decision must be explored. Two contributions are made in this paper. The first is to explore the potential of two attribution methods to explain the output of a model. Attribution works by finding the part of a model's input most responsible for the output. The attribution methods used in the experiments are extremal perturbation and gradient-weighted class activation mapping (GRAD-CAM). The dataset used is Patch Camelyon (P-CAM). The second contribution is to show that relevant image information can be extracted without additional annotated segmentation data. Both of these are important for increasing the interpretability and explainable power of deep networks, ultimately to create models more suitable for clinical use.  
