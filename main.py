

import os

import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from architectures import SimpleCNN
#from datasets import CIFAR10, RotatedImages
from utils import plot

from datasets import PathDataset,ImageDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision
from datetime import datetime

#datadir = "C:/Users/hecto/Documents/ai/python/python2/assignment2/exercise5/training/000"
#datadir = "C:/Users/hecto/Documents/ai/python/python2/assignment2/exercise5/pseudo_train"
#Define the path to the training,validation and test data. Data will be split up later.
#datadir = "C:/Users/hecto/Documents/ai/python/python2/assignment2/exercise5/training"
datadir = "C:/Users/Hector Auvinen/Documents/image_inpainting/training"

def evaluate_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn, device: torch.device):
    """Function for evaluation of a model `model` on the data in `dataloader` on device `device`,
    using the specified `loss_fn` loss function"""
    model.eval()
    # We will accumulate the mean loss in variable `loss`
    loss = 0
    with torch.no_grad():  # We do not need gradients for evaluation
        # Loop over all samples in `dataloader`
        for data in tqdm(dataloader, desc="scoring", position=0):
            # Get a sample and move inputs and targets to device
            inputs, targets, file_names = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Get outputs of the specified model
            outputs = model(inputs)
            
            # Here we could clamp the outputs to the minimum and maximum values of inputs for better performance
            #outputs = outputs * 255


            # Add the current loss, which is the mean loss over all minibatch samples
            # (unless explicitly otherwise specified when creating the loss function!)
            loss += loss_fn(outputs, targets).item()
    # Get final mean loss by dividing by the number of minibatch iterations (which
    # we summed up in the above loop)
    loss /= len(dataloader)
    model.train()
    return loss


def main(results_path, network_config: dict, learningrate: int = 1e-3, weight_decay: float = 1e-5,
         n_updates: int = 100, device: torch.device = torch.device("cuda:0")):
    """Main function that takes hyperparameters and performs training and evaluation of model"""
    # Set a known random seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)
    
    # Prepare a path to plot to
    plotpath = os.path.join(results_path, "plots")
    os.makedirs(plotpath, exist_ok=True)
    
    #
    #image_dataset = ImageDataset(datadir)
    #trainloader = DataLoader(image_dataset, batch_size=1,shuffle=True)
    ###############################################################################
    #
    path_dataset = PathDataset(datadir) #The "first" class (returns the path(s) and index)
    #trainset = ImageDataset(path_dataset)
    #trainloader = DataLoader(trainset, batch_size=1)
    # Split dataset into training, validation and test set

    trainingset = torch.utils.data.Subset(
        path_dataset,
        indices=np.arange(int(len(path_dataset) * (3 / 5))))
    validationset = torch.utils.data.Subset(
        path_dataset,
        indices=np.arange(int(len(path_dataset) * (3 / 5)), int(len(path_dataset) * (4 / 5))))

    testset = torch.utils.data.Subset(
        path_dataset,
        indices=np.arange(int(len(path_dataset) * (4 / 5)), len(path_dataset)))


    # Create datasets and dataloaders with rotated targets without augmentation (for evaluation)
    trainingset_eval = ImageDataset(dataset=trainingset)
    validationset = ImageDataset(dataset=validationset)
    testset = ImageDataset(dataset=testset)

    trainloader = torch.utils.data.DataLoader(trainingset_eval, batch_size=64, shuffle=True, num_workers=2, pin_memory = True)

    valloader = torch.utils.data.DataLoader(validationset, batch_size=1, shuffle=False, num_workers=2, pin_memory = True)

    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2, pin_memory = True)


    ###############################################################################
    
    # Define a tensorboard summary writer that writes to directory "results_path/tensorboard"
    writer = SummaryWriter(log_dir=os.path.join(results_path, "tensorboard"))
    
    # Create Network
    net = SimpleCNN(**network_config)
    net.to(device)
    
    # Get mse loss function
    mse = torch.nn.MSELoss()
    
    # Get adam optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learningrate, weight_decay=weight_decay)
    
    print_stats_at = 100  # print status to tensorboard every x updates
    plot_at = 100  # plot every x updates 10_000
    validate_at = 5000  # evaluate model on validation set and check for new best model every x updates
    update = 0  # current update counter
    best_validation_loss = np.inf  # best validation loss so far
    update_progress_bar = tqdm(total=n_updates, desc=f"loss: {np.nan:7.5f}", position=0)

    # Save initial model as "best" model (will be overwritten later)
    saved_model_file = os.path.join(results_path, "best_model.pt")
    torch.save(net, saved_model_file)

    #now = datetime.now()
    #saved_model_file = os.path.join(results_path,str(now.strftime("%m_%d_%Y_%H_%M_%S")) + "_model.pt")
    #torch.save(net, saved_model_file)
    
    # Train until n_updates updates have been reached
    while update < n_updates:
        for data in trainloader:
            # Get next samples
            #inputs, targets, ids = data
            inputs, targets, ids = data
            #print(inputs.shape)
            #print(targets.shape)
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Reset gradients
            optimizer.zero_grad()
            
            # Get outputs of our network
            outputs = net(inputs)
            
            # Calculate loss, do backward pass and update weights
            loss = mse(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # Print current status and score
            if (update + 1) % print_stats_at == 0:
                writer.add_scalar(tag="training/loss", scalar_value=loss.cpu(), global_step=update)
            
            # Plot output
            if (update + 1) % plot_at == 0:
                plot(inputs.detach().cpu().numpy(), targets.detach().cpu().numpy(), outputs.detach().cpu().numpy(),
                     plotpath, update)

            # Evaluate model on validation set
            if (update + 1) % validate_at == 0:
                val_loss = evaluate_model(net, dataloader=valloader, loss_fn=mse, device=device)
                writer.add_scalar(tag="validation/loss", scalar_value=val_loss, global_step=update)
                # Add weights and gradients as arrays to tensorboard
                for i, (name, param) in enumerate(net.named_parameters()):
                    writer.add_histogram(tag=f"validation/param_{i} ({name})", values=param.cpu(), global_step=update)
                    writer.add_histogram(tag=f"validation/gradients_{i} ({name})", values=param.grad.cpu(), global_step=update)
                # Save best model for early stopping
                if val_loss < best_validation_loss:
                    best_validation_loss = val_loss
                    torch.save(net, saved_model_file)

            update_progress_bar.set_description(f"loss: {loss:7.5f}", refresh=True)
            update_progress_bar.update()
            
            # Increment update counter, exit if maximum number of updates is reached
            # Here, we could apply some early stopping heuristic and also exit if its
            # stopping criterion is met
            update += 1
            if update >= n_updates:
                break

    update_progress_bar.close()
    writer.close()
    print("Finished Training!")
    
    # Load best model and compute score on test set
    print(f"Computing scores for best model")
    net = torch.load(saved_model_file)
    train_loss = evaluate_model(net, dataloader=trainloader, loss_fn=mse, device=device)

    val_loss = evaluate_model(net, dataloader=valloader, loss_fn=mse, device=device)
  
    test_loss = evaluate_model(net, dataloader=testloader, loss_fn=mse, device=device)


    print(f"Scores:")
    print(f"  training loss: {train_loss}")

    print(f"validation loss: {val_loss}")

    print(f"      test loss: {test_loss}")

    
    # Write result to file
    with open(os.path.join(results_path, "results.txt"), "w") as rf:
        print(f"Scores:", file=rf)
        print(f"  training loss: {train_loss}", file=rf)
    
        print(f"validation loss: {val_loss}", file=rf)
          
        print(f"      test loss: {test_loss}", file=rf)



if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Path to JSON config file")
    args = parser.parse_args()
    
    with open(args.config_file) as cf:
        config = json.load(cf)
    main(**config)
