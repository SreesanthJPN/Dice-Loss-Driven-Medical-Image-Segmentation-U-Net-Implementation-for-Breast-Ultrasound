# Breast Ultrasound Image Segmentation Using U-Net and Dice Loss

This project focuses on the segmentation of breast ultrasound images using the U-Net architecture with Dice Loss. It involves deep learning techniques to accurately predict and segment tumor regions from ultrasound scans.

## Introduction

The goal of this project is to apply deep learning techniques to the medical imaging domain, specifically in breast ultrasound segmentation. The U-Net architecture is used due to its efficiency in medical image segmentation tasks. The Dice Loss function is employed to improve the overlap between predicted segmentation masks and ground truth.

## Dataset

The dataset used in this project is the **Breast Ultrasound Images Dataset (BUSI)**, which consists of ultrasound images of breast tumors and their corresponding ground truth masks. The dataset includes three categories: benign, malignant, and normal cases. 

You can find the dataset in : [https://www.kaggle.com/datasets/sabahesaraki/breast-ultrasound-images-dataset]

### Dataset Structure

- **Images**: Breast ultrasound scans.
- **Masks**: Corresponding binary masks indicating tumor regions.

The segmentation model is based on the **U-Net** architecture, a convolutional neural network designed for biomedical image segmentation. U-Net consists of an encoder-decoder structure with skip connections, allowing the model to capture both context and fine-grained details.

### U-Net Layers:
1. **Encoder**: Extracts features from input images at multiple scales.
2. **Bottleneck**: Connects the encoder and decoder.
3. **Decoder**: Reconstructs the segmentation mask from the encoded features.
4. **Output Layer**: A single-channel convolution layer with a sigmoid activation function for binary mask prediction.

‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎![image](https://github.com/user-attachments/assets/d4cd9792-ae84-4189-8d40-2b3ca0a73d17)



## Loss Function

The **Dice Loss** function is used to evaluate the accuracy of the segmentation. Dice Loss is calculated based on the overlap between the predicted segmentation mask and the ground truth mask. It is particularly effective for tasks with imbalanced classes like medical image segmentation.
‎ ‎ ‎ ‎ ‎ 
‎ ‎ ‎ 
‎ ‎ ‎ 

‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎‎![image](https://github.com/user-attachments/assets/546f2d15-0090-4ec2-b247-73a44a12d1d5)
‎ ‎ 
‎ 
