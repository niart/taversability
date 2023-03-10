# Import libraries
import torch
import torch.nn as nn
from torch import optim
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split, Subset

from tqdm import tqdm

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import time

import cv2


# Define the data to be used
# DATASET = "../../datasets/dataset_to_delete/"
DATASET = "dataset_3+8bags_3var3sc_regression_classification_kmeans_split/"


class TraversabilityDataset(Dataset):
    """Custom Dataset class to represent our dataset
    It includes data and information about the data

    Args:
        Dataset (class): Abstract class which represents a dataset
    """
    
    def __init__(self, traversal_costs_file, images_directory,
                 transform=None):
        """Constructor of the class

        Args:
            traversal_costs_file (string): Path to the csv file which contains
            images index and their associated traversal cost
            images_directory (string): Directory with all the images
            transform (callable, optional): Transforms to be applied on a
            sample. Defaults to None.
        """
        # Read the csv file
        self.traversal_costs_frame = pd.read_csv(traversal_costs_file)
        
        # Initialize the name of the images directory
        self.images_directory = images_directory
        
        # Initialize the transforms
        self.transform = transform

    def __len__(self):
        """Return the size of the dataset

        Returns:
            int: Number of samples
        """
        # Count the number of files in the image directory
        # return len(os.listdir(self.images_directory))
        return len(self.traversal_costs_frame)

    def __getitem__(self, idx):
        """Allow to access a sample by its index

        Args:
            idx (int): Index of a sample

        Returns:
            list: Sample at index idx
            ([image, traversal_cost])
        """
        # Get the image name at index idx
        image_name = os.path.join(self.images_directory,
                                  self.traversal_costs_frame.loc[idx, "image_id"])
        
        # Read the image
        image = Image.open(image_name)
        
        # Eventually apply transforms to the image
        if self.transform:
            image = self.transform(image)
        
        # Get the corresponding traversal cost
        traversal_cost = self.traversal_costs_frame.loc[idx, "traversal_cost"]
        
        # Get the corresponding traversability label
        traversability_label = self.traversal_costs_frame.loc[idx, "traversability_label"]

        return image, traversal_cost, traversability_label


mean = torch.tensor([0.3426, 0.3569, 0.2914])
std = torch.tensor([0.1363, 0.1248, 0.1302])

# Define a different set of transforms testing
# (for instance we do not need to flip the image)
test_transform = transforms.Compose([
    # transforms.Resize(100),
    transforms.Resize((70, 210)),
    # transforms.Grayscale(),
    # transforms.CenterCrop(100),
    # transforms.RandomCrop(100),
    transforms.ToTensor(),
    
    # Mean and standard deviation were pre-computed on the training data
    # (on the ImageNet dataset)
    transforms.Normalize(
        mean=mean,
        std=std,
        # mean=[0.485, 0.456, 0.406],
        # std=[0.229, 0.224, 0.225],
    ),
])

# Create a Dataset for testing
test_set = TraversabilityDataset(
    traversal_costs_file=DATASET+"traversal_costs_test.csv",
    images_directory=DATASET+"images_test",
    transform=test_transform
)

BATCH_SIZE = 16

test_loader = DataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    shuffle=False,  # SequentialSampler
    num_workers=4,
    # num_workers=12,
    pin_memory=True,
)

# Use a GPU if available
device = "cpu"
#device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}\n")


# Load the pre-trained model
#model = models.resnet18().to(device=device)
#model = models.mobilenet_v3_small().to(device=device)
model = models.mobilenet_v2().to(device=device)

# Replace the last layer by a fully-connected one with 1 output
#model.fc = nn.Linear(model.fc.in_features, 1, device=device)
#model.classifier[3] = nn.Linear(model.classifier[3].in_features, 1).to(device=device)
model.classifier[1] = nn.Linear(model.last_channel, 1).to(device=device)

# Load the fine-tuned weights
#model.load_state_dict(torch.load("resnet18_fine_tuned_3+8bags_3var3sc_transforms.params", map_location=device))
#model.load_state_dict(torch.load("mobilenet_v3_small_mobilenet.params", map_location=device))
model.load_state_dict(torch.load("v2_mobilenet.params", map_location=device))

# Define the loss function
criterion = nn.MSELoss()

# Testing
test_loss = 0.

# Configure the model for testing
model.eval()

times = []

with torch.no_grad():
    # Loop over the testing batches
    for images, traversal_costs, _ in test_loader:
        loading_start = time.time()
        images = images.to(device)
        traversal_costs = traversal_costs.to(device)
        
        loading_stop = time.time()
        #print("Loading: ", loading_stop - loading_start)
        
        prediction_start = time.time()

        # Perform forward pass
        predicted_traversal_costs = model(images)

        prediction_stop = time.time()
        #print("Prediction: ", prediction_stop - prediction_start)

        times.append(prediction_stop - loading_start)
        
        # Compute loss
        loss = criterion(predicted_traversal_costs[:, 0], traversal_costs)
        
        # Accumulate batch loss to average of the entire testing set
        test_loss += loss.item()


print("Total time: ", np.mean(times[1:-1]))

        
# Compute the loss and accuracy
test_loss /= len(test_loader)

print("Test loss: ", test_loss)
