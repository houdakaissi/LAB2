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



2. ViT Architecture
   
This tutorial provides a comprehensive guide to implementing the Vision Transformer (VIT) model for image classification tasks, specifically using the MNIST dataset. MNIST is a well-known dataset consisting of hand-written digits ranging from 0 to 9, each represented by grayscale images with a resolution of 28x28 pixels. 
STEPS:
tep 1: Preparation

Import the necessary libraries including tqdm for monitoring training progress, numpy, and modules from the PyTorch library. These modules include torch for core functionality, torch.nn for neural network-related tasks, and torch.optim for optimization algorithms like Adam. We'll also utilize CrossEntropyLoss as the loss function for multi-class classification tasks and DataLoader for loading data batches during training. Additionally, ToTensor from torchvision transforms PIL images or NumPy arrays to PyTorch tensors. The MNIST dataset, comprising hand-written digits, will be used for training and testing.

Step 2: Model Initialization and Training

Prepare the model for training on the MNIST dataset with GPU acceleration and a batch size of 128 samples. Initialize the Adam optimizer and the cross-entropy loss function. Train the model iteratively, displaying the loss and epoch progress, and evaluate the test loss and accuracy.

Step 3: Autograd and Backpropagation

Leverage PyTorch's autograd functionality to facilitate neural network training via backpropagation. This involves initial feature extraction, flattening, and embedding, followed by subsequent transformation through the transformer encoder. The encoder consists of several layers, including layer normalization, multi-head self-attention, and multi-layer perceptron layers, orchestrating the transformation of image representations.

Step 4: Patchification

Since the transformer encoder was originally designed for sequence data, it must be modified to handle images. Implement patchification, dividing each image into 49 patches, with each patch covering a 4x4 pixel area.

Step 5: Adding Classification Token

Insert a classification token into the model to capture information about the other tokens.

Step 6: Positional Encoding

Incorporate positional encoding to enable the model to understand the placement of each patch in the original image sequence.

Step 7: Encoder Block (Part 1/2)

The encoder block is constructed, starting with layer normalization to normalize layer activations, followed by multi-head self-attention to allow the model to focus on relevant input parts and capture their relationships.

Step 8: Inserting Encoder Block

Insert the encoder block into the ViT model responsible for patchification before the transformer blocks.

Step 9: Classification MLP

Extract the classification token from each sequence and utilize it for classification, resulting in N classifications per token. The model's output is now an (N, 10) tensor
RESULT: this architecture represents a ViT model for vision tasks, where the input image patches are linearly embedded, processed through transformer blocks, and finally classified using an MLP head
<img width="393" alt="1" src="https://github.com/houdakaissi/LAB2/assets/95725016/aef402e3-de39-4e4e-bb0c-fe14df90e7bf">

2. Model Comparison
Interpreted the results obtained from the ViT model and compared them with the results from the CNN and Faster R-CNN models.




