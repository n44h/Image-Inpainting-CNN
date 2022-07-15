import os
import json
import numpy as np
from tqdm import tqdm
import dill as pkl

import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from architecture import ImageInpaintingCNN
from datasets import RawDataset, InpaintingDataset
from utils import mse, evaluate_model, plot, denormalize_image
from utils import Logger, EarlyStopping


# Parent directory absolute path.
PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Path to the config JSON file.
CONFIG_FILEPATH = os.path.join(PARENT_DIR, "config.json")

# Path to pickled datasets folder.
PKL_PATH = os.path.join(PARENT_DIR, "pkl")
# Pickled training set file.
TRAIN_PKL_PATH = os.path.join(PKL_PATH, "train_set.pkl")
# Pickled validation set file.
VALIDATION_PKL_PATH = os.path.join(PKL_PATH, "eval_set.pkl")
# Pickled test set file.
TEST_PKL_PATH = os.path.join(PKL_PATH, "test_set.pkl")


def main(logfile_path, saved_model_path, results_path,
         training_path, validation_path, test_path,
         network_config: dict, learning_rate: int = 1e-3, weight_decay: float = 1e-5,
         batch_size: int = 40, num_epochs: int = 50_000,
         device: torch.device = torch.device("cuda:0")):

    # Initialize Logger.
    logger = Logger(logfile_path=logfile_path)
    logger.log("Initialized Logger.\nExecuting main.py")

    # Set a known random seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    # Prepare a path to store plots.
    plot_path = os.path.join(results_path, "plots")
    os.makedirs(plot_path, exist_ok=True)

    """ --- Dataset Preparation --- """

    """ Train Inpainting Dataset """
    # If pickled train dataset does not exist, create and pickle it.
    if not os.path.exists(TRAIN_PKL_PATH):
        # Load the raw datasets for training.
        training_raw_set = RawDataset(dir_path=training_path)

        # Create InpaintingDataset datasets for training.
        training_set = InpaintingDataset(dataset=training_raw_set)

        # Pickle the train_set and dump to a pkl file.
        with open(TRAIN_PKL_PATH, 'wb') as f:
            pkl.dump(training_set, f)

    # if the pickled training dataset already exists, load it.
    else:
        with open(TRAIN_PKL_PATH, 'rb') as f:
            training_set = pkl.load(f)

    """ Validation Inpainting Dataset """
    # If pickled validation dataset does not exist, create and pickle it.
    if not os.path.exists(VALIDATION_PKL_PATH):
        # Load the raw datasets for validation.
        validation_raw_set = RawDataset(dir_path=validation_path)

        # Create InpaintingDataset datasets for validation.
        validation_set = InpaintingDataset(dataset=validation_raw_set)

        # Pickle the validation_set and dump to a pkl file.
        with open(VALIDATION_PKL_PATH, 'wb') as f:
            pkl.dump(validation_set, f)

    # if the pickled validation dataset already exists, load it.
    else:
        with open(VALIDATION_PKL_PATH, 'rb') as f:
            validation_set = pkl.load(f)

    """ Test Inpainting Dataset """
    # If pickled test dataset does not exist, create and pickle it.
    if not os.path.exists(TEST_PKL_PATH):
        # Load the raw datasets for test.
        test_raw_set = RawDataset(dir_path=test_path)

        # Create InpaintingDataset datasets for testing.
        test_set = InpaintingDataset(dataset=test_raw_set)

        # Pickle the test_set and dump to a pkl file.
        with open(TEST_PKL_PATH, 'wb') as f:
            pkl.dump(test_set, f)

    # if the pickled test dataset already exists, load it.
    else:
        with open(TEST_PKL_PATH, 'rb') as f:
            test_set = pkl.load(f)

    """ --- End of Dataset Preparation --- """

    # Store the mean and std to denormalize output.
    mean, std = torch.from_numpy(training_set.mean), torch.from_numpy(training_set.std)

    # Create DataLoaders.
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=0)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    # Define a tensorboard summary writer that writes to directory "results_path/tensorboard"
    writer = SummaryWriter(log_dir=os.path.join(results_path, "tensorboard"))

    # Create Network.
    model = ImageInpaintingCNN(**network_config)
    # Send model parameters to appropriate device memory.
    model.to(device)

    # Get adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Plotting settings.
    print_stats_at = 100  # print status to tensorboard every x updates
    plot_at = 10_000  # plot every x updates
    validate_at = 5000  # evaluate model on validation set and check for new best model every x updates
    epoch = 0  # current update counter
    best_validation_loss = np.inf  # best validation loss so far
    update_progress_bar = tqdm(total=num_epochs, desc=f"loss: {np.nan:7.5f}", position=0)

    # Save initial model as "best" model (will be overwritten later).
    saved_model_file = os.path.join(saved_model_path, "best_model.pt")
    torch.save(model, saved_model_file)

    # Create EarlyStopping object.
    early_stopping = EarlyStopping(tolerance=5, min_delta=1)
    val_loss = 0

    # Info message.
    logger.log("Starting training.")

    # Train until num_epochs have been reached
    while epoch < num_epochs:
        # Printing update messages.
        logger.log(f"Currently in Epoch: {epoch}")

        for data in training_loader:
            # Get next samples
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Reset gradients
            optimizer.zero_grad()

            # Get outputs of our network
            outputs = model(inputs.float())

            _batch_size = outputs.size()[0]

            # Stack them on themselves N times, where N is the batch size.
            _mean = torch.reshape(mean, (3, 1))
            _mean = _mean.repeat(_batch_size, 100 * 100)
            _mean = torch.reshape(_mean, (_batch_size, 3, 100, 100))
            _mean = _mean.to(device)

            _std = torch.reshape(std, (3, 1))
            _std = _std.repeat(_batch_size, 100 * 100)
            _std = torch.reshape(_std, (_batch_size, 3, 100, 100))
            _std = _std.to(device)

            # Denormalize output
            outputs = denormalize_image(outputs, _mean, _std)

            # Calculate loss, do backward pass and update weights
            loss = mse(outputs, targets.float())
            loss.backward()
            optimizer.step()

            # Print current status and score
            if (epoch + 1) % print_stats_at == 0:
                writer.add_scalar(tag="training/loss", scalar_value=loss.cpu(), global_step=epoch)

            # Plot output
            if (epoch + 1) % plot_at == 0:
                plot(inputs.detach().cpu().numpy(), targets.detach().cpu().numpy(), outputs.detach().cpu().numpy(),
                     plot_path, epoch)

            # Evaluate model on validation set
            if (epoch + 1) % validate_at == 0:
                val_loss = evaluate_model(model, dataloader=validation_loader, loss_fn=mse, device=device)
                writer.add_scalar(tag="validation/loss", scalar_value=val_loss, global_step=epoch)
                # Add weights and gradients as arrays to tensorboard
                for i, (name, param) in enumerate(model.named_parameters()):
                    writer.add_histogram(tag=f"validation/param_{i} ({name})", values=param.cpu(), global_step=epoch)
                    writer.add_histogram(tag=f"validation/gradients_{i} ({name})", values=param.grad.cpu(),
                                         global_step=epoch)
                # Save best model for early stopping
                if val_loss < best_validation_loss:
                    best_validation_loss = val_loss
                    torch.save(model, saved_model_file)

            update_progress_bar.set_description(f"loss: {loss:7.5f}", refresh=True)
            update_progress_bar.update()

            # Increment update counter, exit if maximum number of updates is reached
            epoch += 1
            if epoch >= num_epochs:
                break

            # Early stopping mechanism.
            early_stopping(train_loss=loss, validation_loss=val_loss)
            if early_stopping.early_stop:
                logger.log(f"Early stopping at epoch: {epoch}")
                break

    update_progress_bar.close()
    writer.close()
    logger.log("Finished Training!")

    # Load best model and compute score on test set
    logger.log(f"Computing scores for best model")
    model = torch.load(saved_model_file)
    train_loss = evaluate_model(model, dataloader=training_loader, loss_fn=mse, device=device)
    val_loss = evaluate_model(model, dataloader=validation_loader, loss_fn=mse, device=device)
    test_loss = evaluate_model(model, dataloader=test_loader, loss_fn=mse, device=device)

    logger.log(f"Scores:")
    logger.log(f"  training loss: {train_loss}")
    logger.log(f"validation loss: {val_loss}")
    logger.log(f"      test loss: {test_loss}")

    # Write result to file
    with open(os.path.join(saved_model_path, "results.txt"), "w") as rf:
        print(f"Scores:", file=rf)
        print(f"  training loss: {train_loss}", file=rf)
        print(f"validation loss: {val_loss}", file=rf)
        print(f"      test loss: {test_loss}", file=rf)


if __name__ == "__main__":
    # Parsing the config file.
    with open(CONFIG_FILEPATH, "r") as conf:
        CONFIG = json.load(conf)

    # Get the training configurations.
    TRAINING_CONFIG = CONFIG["training_config"]

    # Get the network configurations.
    NETWORK_CONFIG = CONFIG["network_config"]

    # Device: GPU or CPU.
    available_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Dict of paths.
    PATHS = {"results_path": CONFIG["results_path"],
             "saved_models_path": CONFIG["saved_models_path"],
             "pkl_path": CONFIG["pkl_dir_path"],
             "log_path": CONFIG["log_path"]
             }
    # If the folders don't exist, create them.
    for key, path in PATHS.items():
        # Construct absolute path.
        path = os.path.join(PARENT_DIR, path)

        # Save the absolute paths.
        PATHS[key] = path

        # Create the dirs if they do not exist.
        os.makedirs(path, exist_ok=True)

    # Construct logfile path.
    logfile_path = os.path.join(PATHS["log_path"], "logfile.log")

    # Call main().
    main(logfile_path=logfile_path,
         saved_model_path=PATHS["saved_models_path"],
         results_path=PATHS["results_path"],
         training_path=os.path.join(*CONFIG["train_dataset_path"]),
         validation_path=os.path.join(*CONFIG["eval_dataset_path"]),
         test_path=os.path.join(*CONFIG["test_dataset_path"]),
         network_config=NETWORK_CONFIG,
         learning_rate=TRAINING_CONFIG["learning_rate"],
         weight_decay=TRAINING_CONFIG["weight_decay"],
         batch_size=TRAINING_CONFIG["batch_size"],
         num_epochs=TRAINING_CONFIG["num_epochs"],
         device=available_device)
