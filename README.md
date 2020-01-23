# AFANet

  Implemention of Paper：Deep Adaptive Feature Aggregation in Multi-task Convolutional Neural Networks   
  
## Introduction
  ![](https://github.com/IJCAI2020-MTL/AFANet/raw/master/Architecture.png)   
  The Adaptive Feature Aggregation (AFA) layer for multi-task CNNs, in which a dynamic aggregation mechanism is designed to allow each task adaptively determines the degree to which the feature aggregation of different tasks is needed. We introduce two types of aggregation modules to the AFA layer, which realize the adaptive feature aggregation by capturing the feature dependencies along the channel and spatial axes, respectively.   
  
  | 表格      | 第一列     | 第二列     |
| ---------- | :-----------:  | :-----------: |
| 第一行     | 第一列     | 第二列     |
  
## Requirements  

  Python >= 3.6  
  numpy  
  PyTorch >= 1.0  
  torchvision >= 0.2   
  tensorboardX  
  
## Installation
  1. Clone the repo:   
    ```
    git clone https://github.com/IJCAI2020-MTL/AFANet.git   
    ```   
    ```
    cd AFANet
    ```
  2. For custom dependencies:   
    ```
    pip install matplotlib tensorboardX   
    ```

## For Training   
  1. Configure your dataset path in `Dataloader.py`.   
  2. Deeplabv3+ networks with ResNet backbone are used to train semantic segmentation and depth prediction(see full input arguments in    ```train_multi.py``` ):   
    ```
    python train_multi.py
    ```


