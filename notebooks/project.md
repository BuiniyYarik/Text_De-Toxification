# CAM and SeCAM: Explainable AI for Understanding Image Classification Models

## Introduction
Explainable Artificial Intelligence (XAI) aims to make the results of AI models understandable to humans. 
This is crucial in many applications where understanding the basis of an AI's decision is as important as the decision itself. 
In this tutorial, we will focus on Class Activation Mapping (CAM) and Segmentation Class Activation Mapping (SeCAM), 
XAI methods applied to the Convolutional Neural Networks solving image classification problems. 
Particularly, we will use ResNet50 model, one of the most popular model in image classification.

## Section 1: Overview of XAI Methods
Explainable AI (XAI) refers to methods and techniques in the application of artificial intelligence technology such that the results of the solution can be understood by human experts. It contrasts with the concept of the "black box" in machine learning where even their designers cannot explain why the AI arrived at a specific decision. XAI is becoming increasingly important as AI systems are used in more critical applications such as diagnostic healthcare, autonomous driving, and more.

## Section 2: ResNet50 Architecture and Importance of Understanding


## Section 3: Class Activation Mapping (CAM)
# Class Activation Mapping (CAM)

## General Definition

Class Activation Mapping (CAM) is a technique used to identify the discriminative regions in an image that contribute to the class prediction made by a Convolutional Neural Network (CNN). CAM is particularly useful for understanding and interpreting the decisions of CNN models.

## How CAM Works

### Steps Involved in CAM:

1. **Feature Extraction**:
    - Extract the feature maps from the last convolutional layer of the CNN.

2. **Global Average Pooling (GAP)**:
    - Apply Global Average Pooling (GAP) to the feature maps to get a vector of size equal to the number of feature maps.

3. **Fully Connected Layer**:
    - The GAP output is fed into a fully connected layer to get the final class scores.

4. **Class Activation Mapping**:
    - For a given class, compute the weighted sum of the feature maps using the weights from the fully connected layer.

### Equations

1. **Feature Maps**:
    - Let ( $f_k(x, y)$ ) represent the activation of unit \( k \) in the feature map at spatial location \( (x, y) \).

2. **Global Average Pooling**:
    - The GAP for feature map \( k \) is computed as:
      \[
      F_k = \frac{1}{Z} \sum_{x} \sum_{y} f_k(x, y)
      \]
      where \( Z \) is the number of pixels in the feature map.

3. **Class Score**:
    - The class score \( S_c \) for class \( c \) is computed as:
      \[
      S_c = \sum_{k} w_{k}^{c} F_k
      \]
      where \( w_{k}^{c} \) is the weight corresponding to class \( c \) for feature map \( k \).

4. **Class Activation Map**:
    - The CAM for class \( c \) is computed as:
      \[
      M_c(x, y) = \sum_{k} w_{k}^{c} f_k(x, y)
      \]
      This gives the importance of each spatial element \( (x, y) \) in the feature maps for class \( c \).
$$
### Example

Let's consider an example where we use a pre-trained ResNet50 model to generate CAM for a given image.

```python
import numpy as np
import cv2
from torchvision import models, transforms
import torch.nn.functional as F
from torch import topk
import matplotlib.pyplot as plt

# Load the pre-trained ResNet50 model
model = models.resnet50(pretrained=True)
model.eval()

# Hook the feature extractor
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

model._modules.get('layer4').register_forward_hook(hook_feature)

# Load and preprocess the image
def preprocess_image(img_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess(img).unsqueeze(0)
    return img

image_tensor = preprocess_image('input/dogs.jpg')

# Forward pass
outputs = model(image_tensor)

# Get the class indices of top k probabilities
probabilities = F.softmax(outputs, dim=1).data.squeeze()
class_idx = topk(probabilities, 1)[1].int()

# Get the softmax weights
params = list(model.parameters())
softmax_weights = np.squeeze(params[-2].data.numpy())

# Generate CAM
def generate_CAM(feature_maps, softmax_weights, class_indices):
    batch_size, num_channels, height, width = feature_maps.shape
    output_cams = []

    for class_idx in class_indices:
        cam = softmax_weights[class_idx].dot(feature_maps.reshape((num_channels, height * width)))
        cam = cam.reshape(height, width)
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        cam_img = np.uint8(255 * cam)
        output_cams.append(cam_img)

    return output_cams

CAMs = generate_CAM(features_blobs[0], softmax_weights, class_idx)

# Display and save the CAM results
def display_and_save_CAM(CAMs, width, height, original_image, class_indices, class_labels, save_name):
    for i, cam in enumerate(CAMs):
        heatmap = cv2.applyColorMap(cv2.resize(cam, (width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.3 + original_image * 0.5
        cv2.putText(result, class_labels[class_indices[i]], (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        cv2.imwrite(f"outputs/CAM_{save_name}_{i}.jpg", result)

class_labels = load_class_labels('LOC_synset_mapping.txt')
save_name = 'dogs'
original_image = cv2.imread('input/dogs.jpg')
display_and_save_CAM(CAMs, original_image.shape[1], original_image.shape[0], original_image, class_idx, class_labels, save_name)


## Section 3: Segmentation Class Activation Mapping (SeCAM)