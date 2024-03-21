# Deep Learning Lab Report
#Objective    
The main objective of this lab is to gain familiarity with the PyTorch library and build various neural architectures for computer vision tasks. Specifically, we aim to implement CNN, Faster R-CNN, and Vision Transformer (ViT) models for classifying the MNIST dataset.
Part 1: CNN Classifier
 
<img width="82" alt="Capture d'écran 2024-03-20 170202" src="https://github.com/houdakaissi/LAB2/assets/95725016/99352a87-a004-43f0-9d37-030985c053a3">
1. CNN Architecture
Implemented a CNN architecture using PyTorch to classify the MNIST dataset.
Defined convolutional, pooling, and fully connected layers.
Specified hyperparameters such as kernels, padding, stride, and optimizers.
Ran the model in GPU mode for improved performance.
2. Faster R-CNN
Implemented the Faster R-CNN architecture for MNIST dataset classification.
3. Model Comparison
Compared the performance of the CNN and Faster R-CNN models using metrics such as accuracy, F1 score, loss, and training time.
4. Fine-tuning with VGG16 and AlexNet
Retrained pre-trained models (VGG16 and AlexNet) on the MNIST dataset.
Compared the performance of fine-tuned models with CNN and Faster R-CNN.
Part 2: Vision Transformer (ViT)



1. ViT Architecture
This tutorial provides a comprehensive guide to implementing the Vision Transformer (VIT) model for image classification tasks, specifically using the MNIST dataset. MNIST is a well-known dataset consisting of hand-written digits ranging from 0 to 9, each represented by grayscale images with a resolution of 28x28 pixels. 
steps:
import:tqdm library is commonly used to monitor the training progress of machine learning models ,numpy,modules from the PyTorch library:
torch is the main PyTorch module.
torch.nn contains neural network related functionalities.
torch.optim contains optimization algorithms, with Adam being one of the most commonly used optimizers.
CrossEntropyLoss is a loss function often used for multi-class classification tasks.
DataLoader is used to load data batches during training.ToTensor is a torchvision transform that converts PIL images or NumPy arrays to PyTorch tensors. MNIST is a dataset containing hand-written digits, commonly used for training and testing machine learning models.
step2:t prepares the model for training on the MNIST dataset,with GPU with 128 samples in each batch  Initializes the Adam optimizer,nitializes the cross-entropy loss function  train the model in loops and show loss and epoch and test test loss and accuracy
sytep3:pytroch provides autograd : facilitating the training of neural network models through backpropagation. 
----> the sequence of operations typically involves initial feature extraction :flattening and embedding followed by subsequent transformation through the transformer encoder, which consists of several layers omprising layer normalization, multi-head self-attention, and multi-layer perceptron layers, orchestrates the transformation of image representation
   step4:The transformer encod er was developed with sequence data in mind, an image is not a sequenc so we do We modify our MyViT class to implement the patchifying only
   etp!:each image will be divided into 49 patches, each patch covering an area of 4x4 pixels
   Adding the classification token
Step 2: Adding the classification token we e add to our model that has the role of capturing information about the other tokens.
Step 3: Positional encoding ositional encoding allows the model to understand where each patch would be placed in the original image n each sequence, for token i we add to its j-th coordinate  
Step 4: The encoder block (Part 1/2)The first part of the encoder block applies Layer Normalization:Layer normalization is a technique commonly used in neural networks to normalize the activations of a layer, and Multi-head Self Attention:s to allow the model to focus on relevant parts of the input and capture the relationships between them 

3. Model Comparison
Interpreted the results obtained from the ViT model and compared them with the results from the CNN and Faster R-CNN models.




