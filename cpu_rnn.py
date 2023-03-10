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

transform = transforms.Compose([
    # Reduce the size of the images
    # (if size is an int, the smaller edge of the
    # image will be matched to this number and the ration is kept)
    transforms.Resize((70, 210)),

    # Convert a PIL Image or numpy.ndarray to tensor
    transforms.ToTensor(),
])

def calculate_overall_cost(arr):
    if len(arr) == 1:
        # if only one entry, it takes up 100% weight
        return arr[0]
    elif len(arr) == 2:
        # if only two entries, first entry takes up 70% and second entry takes up 30%
        adjusted_weights = np.array([0.7, 0.3])
        return np.average(arr, weights=adjusted_weights)
    else:
        # if there are more than two entries, apply the custom weights
        if arr[0] >= max(arr[1:]):
            # if the first entry is the largest, it takes up 100% weight
            return arr[0]
        else:
            # apply the custom weights: 50% to first entry, 30% to second entry, and 20% to third entry
            adjusted_weights = np.array([0.5, 0.3, 0.2])
            total_weight = adjusted_weights.sum()
            adjusted_weights = (adjusted_weights / total_weight) * len(arr)
            return np.average(arr, weights=adjusted_weights)
            
class TraversabilityCostDataset(Dataset):
    def __init__(self, data_path, metadata_path, seq_len, transform):
        self.data_path = data_path
        self.metadata_path = metadata_path
        self.seq_len = seq_len

        # Load the metadata
        self.metadata = pd.read_csv(metadata_path)

        # Group the crops by trajectory ID
        self.trajectories = self.metadata.groupby('trajectory_id')['crop_id'].apply(list).to_dict()

        # Sort the trajectory IDs
        self.trajectory_ids = sorted(list(self.trajectories.keys()))

        self.transform = transform

    def __len__(self):
        return len(self.trajectory_ids)

    def __getitem__(self, idx):
        # Get the trajectory ID for the current item
        traj_id = self.trajectory_ids[idx]

        # Load the crops and traversal costs for the current trajectory
        crop_ids = self.trajectories[traj_id][:self.seq_len]
        crops = []
        for crop_id in crop_ids:
            image = Image.open(os.path.join(self.data_path, f'{crop_id:05d}.png')).convert('RGB')
            crops.append(image)

        if self.transform:
            crops = [self.transform(crop) for crop in crops]

        traversal_costs = self.metadata[self.metadata['trajectory_id'] == traj_id].sort_values('image_timestamp')['traversal_cost'].to_numpy()[:self.seq_len]
 
        overall_cost = calculate_overall_cost(traversal_costs)
        # Pad the sequence with zero-filled tensors if it is too short
        if len(crops) < self.seq_len:
            num_padding = self.seq_len - len(crops)
            pad_tensor = torch.zeros_like(crops[0])
            crops += [pad_tensor] * num_padding

        # Return the crops and overall cost as tensors
        return torch.stack(crops), overall_cost




# Use a GPU if available
device = "cpu"
#device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}\n")



class TraversabilityCostGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(TraversabilityCostGRU, self).__init__()
        self.gru1 = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.gru2 = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, _ = self.gru1(x)
        output, _ = self.gru2(output)
        output = self.fc(output[:, -1, :])
        return output



# Define the loss function
criterion = nn.MSELoss()

# Initialize the model

input_size = 70*210*3  # Crop image channels
hidden_size = 256  # LSTM hidden size
num_layers = 2 # Number of LSTM layers
output_size = 1  # Regression output size

model = TraversabilityCostGRU(input_size, hidden_size, num_layers, output_size).to(device)

# Get all the parameters of the new model
base_params = model.parameters()

# Create data loaders
seq_len = 3
test_dataset = TraversabilityCostDataset('./dataset_3+8bags_3var3sc_regression_classification_kmeans_split/images_test', './dataset_3+8bags_3var3sc_regression_classification_kmeans_split/new_test.csv', seq_len=seq_len, transform=transform)


# Define hyperparameters
optimizer = optim.SGD(model.parameters(), lr=1e-3)
batch_size = 16

test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True)


model = TraversabilityCostGRU(input_size, hidden_size, num_layers, output_size).to(device)

model.load_state_dict(torch.load("100_sequence_gru.params", map_location=device))

# Define the loss function
criterion = nn.MSELoss()

# Testing
test_loss = 0.

# Configure the model for testing
model.eval()

times = []

with torch.no_grad():
    # batch_y is the traversal_costs of a batch (32 sequences)
    for batch_x, batch_y in test_loader:
	    loading_start = time.time()
	    batch_y = batch_y.type.to(device)
	    batch_x = batch_x.to(device)
	    loading_stop = time.time()
	    prediction_start = time.time()
	    predicted_traversal_costs = model(batch_x.view(batch_size, seq_len, input_size))
	    prediction_stop = time.time()
	    times.append(prediction_stop - loading_start)
	    loss = nn.MSELoss()(predicted_traversal_costs.squeeze(), batch_y)
	    test_loss += loss.item() * batch_x.size(0)
	    print("Total loss until this batch is", test_loss)

print("Total time: ", np.mean(times[1:-1]))

        
# Compute the loss and accuracy
print("After testing, Total loss is", test_loss)
print("len(test_loader.dataset) is...", len(test_loader.dataset))

test_loss /= len(test_loader.dataset)
print("Test loss: ", test_loss)
