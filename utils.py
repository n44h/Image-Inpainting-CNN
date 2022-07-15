import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Tuple
from matplotlib import pyplot as plt

import torch
import torch.utils.data
from torchvision.transforms import transforms


def mse(outputs, targets):
    mse_loss = torch.nn.MSELoss()

    # Getting just the unknown cells.
    # masked_outputs = outputs[known_arrays < 1]

    # Returning the MSE loss.
    # return mse_loss(masked_outputs, targets)

    return mse_loss(outputs, targets)


def evaluate_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn, device: torch.device):
    # Switch model to evaluation mode.
    model.eval()

    # Variable to accumulate the mean loss
    loss = 0

    # Start a context without gradient calculation. Grads not required during eval.
    with torch.no_grad():
        # Loop over all samples in the evaluation dataloader.
        for data in tqdm(dataloader, desc="scoring", position=0):
            # Get a sample and move inputs and targets to device
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Get outputs of the specified model
            outputs = model(inputs.float())

            # Here we could clamp the outputs to the minimum and maximum values of inputs for better performance

            # Sum up the losses for each minibatch.
            loss += loss_fn(outputs, targets).item()
    # Get final mean loss by dividing by the number of minibatch iterations.
    loss /= len(dataloader)

    # Switch model back to training mode.
    model.train()

    return loss


def plot(inputs, targets, predictions, path, update):
    """Plotting the inputs, targets and predictions to file `path`"""
    os.makedirs(path, exist_ok=True)
    fig, axes = plt.subplots(ncols=3, figsize=(15, 5))

    for i in range(len(inputs)):
        for ax, data, title in zip(axes, [inputs, targets, predictions], ["Input", "Target", "Prediction"]):
            ax.clear()
            ax.set_title(title)
            ax.imshow(data[i, 0], cmap="gray", interpolation="none")
            ax.set_axis_off()
        fig.savefig(os.path.join(path, f"{update:07d}_{i:02d}.png"), dpi=100)

    plt.close(fig)


def crop_image(pil_image: Image, new_height: int = 100, new_width: int = 100):
    resize_transforms = transforms.Compose([
        transforms.Resize(size=(new_height, new_width)),
        transforms.CenterCrop(size=(new_height, new_width)),
    ])

    # Perform the crop and convert to numpy array.
    return resize_transforms(pil_image)


def normalize_image(image, mean, std):
    image -= mean
    image /= std
    return image


def denormalize_image(image, mean, std):
    # Converting from Torch Tensor to numpy array for calculations.
    image *= std
    image += mean
    return image


def generate_input(image_array, offset: Tuple[int, int], spacing: Tuple[int, int]):
    # Check if image array is an instance of numpy array.
    if not isinstance(image_array, np.ndarray):
        raise TypeError(f"image_array: expected type numpy.ndarray; got type {type(image_array)}.")

    # Check if image_array is a 3D array.
    if len(image_array.shape) != 3:
        raise NotImplementedError("image_array should be 3-dimensional;",
                                  f"instead it is {len(image_array.shape)}-dimensional.")

    # Check if the 3rd dimension is equal to 3 (corresponding to the RGB color channels).
    if image_array.shape[2] != 3:
        raise NotImplementedError("3rd dimension size is not 3.")

    # Checking if offset and spacing values are of type int and are in range.
    for i in range(2):
        if type(offset[i]) is not int:
            raise ValueError(f"offset[{i}] is not of type int. Got type {type(offset[i])} instead.")

        if offset[i] < 0 or offset[i] > 32:
            raise ValueError(f"value of offset[{i}] is not in range 0-32.")

        if type(spacing[i]) is not int:
            raise ValueError(f"spacing[{i}] is not of type int. Got type {type(spacing[i])} instead.")

        if spacing[i] < 2 or spacing[i] > 8:
            raise ValueError(f"value of spacing[{i}] is not in range 2-8.")

    im_height = image_array.shape[0]
    im_width = image_array.shape[1]

    # Storing a transposed version of image_array in input_array: (H, W, 3) -> (3, H, W).
    # Shape of input_array = (3, H, W)
    input_array = np.copy(np.transpose(image_array, (2, 0, 1)))

    # Array of same shape as input_array. Stores 1 for known pixels and 0 for unknown pixels.
    # Shape of known_array = (3, H, W)
    known_array = np.zeros_like(input_array)

    # Denoting the cells that should be known in the known_array.
    # Splice the array to get the positions of where the known cells will be and set them to 1.
    # In known_array, 1 denotes known cell and 0 denotes unknown cell.
    known_array[:, offset[1]::spacing[1], offset[0]::spacing[0]] = 1

    # Any cell that is 0 in known_array (i.e. it is unknown), the corresponding cell in input_array should be put to 0.
    # i.e. corresponding cell in input_array is deleted.
    input_array[known_array < 1] = 0

    # As image_array has shape (H, W, 3) and known_array has shape (3, H, W), we need to transpose image_array
    # to have the same shape as known_array so that the boolean mask ([known_array < 1]) can work.
    target_array = np.transpose(image_array, (2, 0, 1))[known_array < 1]

    # Convert to tensors.
    input_array = torch.from_numpy(input_array)
    known_array = torch.from_numpy(known_array)
    target_array = torch.from_numpy(target_array)
    image_array = torch.from_numpy(np.transpose(image_array, (2, 0, 1)))

    # return input_array, known_array, target_array
    return input_array, known_array, image_array


def collate_fn():
    pass


class Logger:
    def __init__(self, logfile_path: str):
        # Parent directory absolute path.
        parent_dir = os.path.dirname(os.path.abspath(__file__))
        # Path to the logfile.
        self.logfile_path = os.path.join(parent_dir, logfile_path)

        # If the logfile does not exist, create it.
        if not os.path.exists(self.logfile_path):
            with open(self.logfile_path, 'w') as f:
                f.write("Inpainting CNN Log file.\n")

    def log(self, message: str):
        with open(self.logfile_path, 'a') as logfile:
            logfile.write(f"{message}\n")


class EarlyStopping:
    def __init__(self, tolerance: int = 5, min_delta: int = 1):
        self.tolerance = tolerance
        self.min_delta = min_delta

        # Keeps count of the number of times that (val_loss - train_loss) > min_delta.
        self.counter = 0

        # Boolean to trigger early stopping.
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter += 1

            # If the counter is outside the tolerance, trigger early stopping of model training.
            if self.counter >= self.tolerance:
                self.early_stop = True
