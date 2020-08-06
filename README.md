# AFANet

  Implementation of Paper：Adaptive Feature Aggregation in Deep Multi-task Convolutional Neural Networks   
  
## Introduction
  The Adaptive Feature Aggregation (AFA) layer for multi-task CNNs, in which a dynamic aggregation mechanism is designed to allow each task adaptively determines the degree to which the feature aggregation of different tasks is needed. We introduce two types of aggregation modules to the AFA layer, which realize the adaptive feature aggregation by capturing the feature dependencies along the channel and spatial axes, respectively.   
  
## Structure
![image](https://github.com/zhenshen-mla/AFANet/blob/master/examples/structure.png)
  Structure of the AFA layer for multi-task CNNs. It is a plug-and-play component to connect any intermediate convolutional layers of single-task CNNs. Inside the AFA layer, there are sequentially two modules, i.e., CAM and SAM, to support the adaptive feature aggregation between tasks. The output feature maps remain the same size as the inputs, and can be directly fed to the next layers of single-task CNNs.   
  
## Models
  * `/models/layer_afa.py`: implementation of afa layer;
  * `/models/net_image_resnet.py`: single task network based resnet50;   
  * `/models/net_image_afalayer.py`: image tasks aggregation with afa layer;   
  * `/models/net_pixel_deeplab.py`: single task network based deeplab;   
  * `/models/net_pixel_afalayer.py`: pixel tasks aggregation with afa layer;   
  
## Discussion
  In multi-task structure, due to the fact that each task has different parameter complexity and convergence speed, some tasks may be dominated by one task in the process of model training, especially in the parameter updating process of back propagation, which may lead to some tasks deviate from their training objectives.  
  
  Our current study mainly concentrates on designing ﬂexible multi-task CNN architectures. And in the experiments, we have noticed that there is a great difference in gradient magnitude and convergence speed between tasks (especially in semantic segmentation and depth prediction). If we jointly train two tasks without any balancing control, the multi-task gradient could be easily dominated by one task gradient, which comes at the cost of degrading the performance of the other task.  
  
   Several recent studies engaged in weighting the relative contributions of each task in the loss function of multi-task CNNs. In this work[1], they used a joint likelihood formulation to learn task weights based on the homoscedastic uncertainty in each task. And in GradNorm[2], author proposed a gradient normalization algorithm that automatically balances training in multi-task models by dynamically tuning gradient magnitudes. 
  
## Requirements  

  Python >= 3.6  
  numpy  
  torch  
  torchvision  
  tensorboardX  
  sklearn  
  

## Installation
  1. Clone the repo:   
    ```
    git clone https://github.com/zhenshen-mla/AFANet.git   
    ```   
    ```
    cd AFANet
    ```
  2. For custom dependencies:   
    ```
    pip install matplotlib tensorboardX sklearn  
    ```
## For Training   
  1. Download the dataset([NYUv2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html), [Adience benckmark](https://talhassner.github.io/home/projects/Adience/Adience-data.html#frontalized)) and configure the data path.   
  2. Train the single-task and save the pretrained single-task model in `/weight`:   
  3. For pixel tasks, using Deeplabv3+ network with ResNet backbone to conduct semantic segmentation and depth prediction. For image tasks, using ResNet network to conduct age prediction and gender classification (load pretrained model in `/weight`):   
  
