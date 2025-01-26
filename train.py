#-----  Command to run from terminal  -----#
# python -u train.py -c config/base_config.yaml
# Documentation command [execute in path "..\doc"]: .\make.bat html


# Standard library imports
import os

# Third-party imports
import torch
import torch.nn as nn
from tqdm import tqdm

# Configuration and utility imports
from yaml_config_override import add_arguments
from addict import Dict

# Local application/library specific imports
from data_classes.manage_dataset import *
from model_classes.resnet_model import ResNet, ResidualBlock
from utils import *


# Train a training epoch for ResNet model
def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    """
    Trains a model for one epoch.

    :param model: The model to be trained.
    :type model: torch.nn.Module
    :param dataloader: The DataLoader for the training dataset.
    :type dataloader: torch.utils.data.DataLoader
    :param criterion: The criterion to use for calculating loss during training.
    :type criterion: torch.nn.modules.loss._Loss
    :param optimizer: The optimizer to use for updating the model parameters.
    :type optimizer: torch.optim.Optimizer
    :param scheduler: The learning rate scheduler.
    :type scheduler: torch.optim.lr_scheduler._LRScheduler
    :param device: The device on which to train the model (e.g., 'cpu', 'cuda').
    :type device: str
    :return: Returns a dictionary containing the performance metrics of the training dataset.
    :rtype: dict
    """

    # Set the model to training mode
    model.train() 
    # Inizialize variables
    running_loss = 0.0
    predictions = []
    references = []
    # Batch execution
    for i, batch in enumerate(tqdm(dataloader, desc='Training')):
        # Upload images and labels
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        # Erasing old gradient values
        optimizer.zero_grad()
        # Calculate output
        outputs = model(images)
        # Calculate the loss through the previously chosen loss function
        loss = criterion(outputs, labels)
        # Calculate the gradient
        loss.backward()
        # Optimization
        optimizer.step()
        # Scheduler learning rate
        scheduler.step()
        # Add the current loss to the total
        running_loss += loss.item()
        # Compute predictions
        pred = torch.argmax(outputs, dim=1)
        predictions.extend(pred.cpu().numpy())
        # Compute refereces
        references.extend(labels.cpu().numpy())
    # Compute performance metrics based on differences between predictiones and references 
    train_metrics = compute_metrics(predictions, references)
    # Add loss to performance metrics
    train_metrics['loss'] = running_loss / len(dataloader)
    # Return performance metrics
    return train_metrics

# Manage best model and best validation metrics
def manage_best_model_and_metrics(model, evaluation_metric, val_metrics, best_val_metric, best_model, lower_is_better):
    """
    Manages the best model and its metrics.

    :param model: The current model.
    :type model: torch.nn.Module
    :param evaluation_metric: The metric against which to determine the best model.
    :type evaluation_metric: str
    :param val_metrics: The validation metrics of the current model.
    :type val_metrics: dict
    :param best_val_metric: The best validation metrics so far.
    :type best_val_metric: dict
    :param best_model: The best model so far.
    :type best_model: torch.nn.Module
    :param lower_is_better: Whether a lower metric value is better.
    :type lower_is_better: bool
    :return: Returns the best validation metrics and the best model.
    :rtype: tuple (dict, torch.nn.Module)
    """

    # Based on the evaluation metric you choose, evaluate whether the current one is higher or not
    if lower_is_better:
        is_best = val_metrics[evaluation_metric] < best_val_metric[evaluation_metric]
    else:
        is_best = val_metrics[evaluation_metric] > best_val_metric[evaluation_metric]
    if is_best:
        print(f"New best ResNet model found with val {evaluation_metric}: {val_metrics[evaluation_metric]:.4f}")
        best_val_metric = val_metrics
        best_model = model
    return best_val_metric, best_model

# Train deep learning models
def train_model(model, config, train_dl, val_dl, device, criterion):
    """
    This function trains a model given a training dataset, a device and a criterion.

    :param model: The model to be trained.
    :type model: torch.nn.Module
    :param config: The configuration for training the model.
    :type config: Config
    :param train_dl: The DataLoader for the training dataset.
    :type train_dl: torch.utils.data.DataLoader
    :param val_dl: The DataLoader for the validation dataset.
    :type val_dl: torch.utils.data.DataLoader
    :param device: The device on which to train the model (e.g. 'cpu' or 'cuda').
    :type device: str
    :param criterion: The criterion to use for calculating loss during training.
    :type criterion: torch.nn.modules.loss._Loss
    :return: Returns the best validation metric, the best model, the list of training metrics and the list of validation metrics.
    :rtype: tuple (dict, torch.nn.Module, list, list)
    """

    # Initializing variables
    training_metrics_list = []
    validation_metrics_list = []
    
    # Select the optimization algorithm based on the configuration chosen in the config
    if config.deep_learning_training.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    elif config.deep_learning_training.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.training.learning_rate)
    else:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config.training.learning_rate)
    
    # learning rate scheduler
    total_steps = len(train_dl) * config.deep_learning_training.epochs
    warmup_steps = int(total_steps * config.deep_learning_training.warmup_ratio)
    # Warmup for [warmup_ratio]% and linear decay
    scheduler_lambda = lambda step: (step / warmup_steps) if step < warmup_steps else max(0.0, (total_steps - step) / (total_steps - warmup_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler_lambda)
    
    # Initialization of the dictionary that contains performance metrics
    best_val_metric = {'accuracy': float, 'precision': float, 'recall': float, 'f1': float}
    # Initializing the best performance metrics and the other variables
    if config.training.best_metric_lower_is_better:
        best_val_metric[config.training.evaluation_metric.lower()] = float('inf')
    else:
        best_val_metric[config.training.evaluation_metric.lower()] = float('-inf')
    best_model = None
    no_valid_epochs = 0
    
    # Training cycle that iterates through all epochs
    for epoch in range(config.deep_learning_training.epochs):
        print("%s Epoch %d/%d" % ("ResNet model -", (epoch + 1), config.deep_learning_training.epochs))
        # Train an epoch and calculate performance on training
        train_metrics = train_one_epoch(model, train_dl, criterion, optimizer, scheduler, device)
        # Add the performance to the list that will contain the history of all the training performance
        training_metrics_list.append(train_metrics)
        # Calculate performance on validation
        val_metrics, conf_matrix = evaluate(model, val_dl, criterion, device)
        # Add the performance to the list that will contain the history of all the validation performance
        validation_metrics_list.append(val_metrics)
        # Print results
        print(f"Train loss: {train_metrics['loss']:.4f} - Train accuracy: {train_metrics['accuracy']:.4f}")
        print(f"Val loss: {val_metrics['loss']:.4f} - Val accuracy: {val_metrics['accuracy']:.4f}")
        # Calculates the new best model and new best performance as a result of the last training
        best_val_metric, best_model = manage_best_model_and_metrics(
            model, 
            config.training.evaluation_metric, 
            val_metrics, 
            best_val_metric, 
            best_model, 
            config.training.best_metric_lower_is_better
        )
        
        # Earling stopping
        # Check which metric was chosen for early stopping
        if config.training.early_stopping_metric.lower() == 'loss':
            # If performance is not improved, the early stopping counter increases by one
            if val_metrics[config.training.early_stopping_metric.lower()] > best_val_metric[config.training.early_stopping_metric.lower()]:
                no_valid_epochs += 1
        else:
            # If performance is not improved, the early stopping counter increases by one
            if val_metrics[config.training.early_stopping_metric.lower()] < best_val_metric[config.training.early_stopping_metric.lower()]:
                no_valid_epochs += 1
        # If you have reached the maximum value chosen through the config file, end the training
        if no_valid_epochs == config.training.earling_stopping_max_no_valid_epochs:
            print(f"The training process has ended because the maximum value of early stopping, which is {config.training.earling_stopping_max_no_valid_epochs:}, has been reached.")
            break
    # Returns the best metrics, best model, and performance history for training and validation
    return best_val_metric, best_model, training_metrics_list, validation_metrics_list

# Evaluate the performance of the model on the incoming passed dataset
def evaluate_model(best_val_metric, best_model, test_set, criterion, device):
    """
    This function evaluates the best model on a test dataset.

    :param best_val_metric: The best validation metric obtained during training.
    :type best_val_metric: dict
    :param best_model: The best model obtained during training.
    :type best_model: torch.nn.Module
    :param test_set: Test dataloader
    :type test_set: torch.utils.data.DataLoader
    :param criterion: The criterion to use for calculating loss during evaluation.
    :type criterion: torch.nn.modules.loss._Loss
    :param device: The device on which to evaluate the model (e.g., 'cpu', 'cuda').
    :type device: str
    :return: Returns the metrics and confusion matrix for the test dataset.
    :rtype: tuple (list, numpy.ndarray)
    """

    print("---------------------")
    print_best_val_metrics(best_val_metric)
    # Evaluate the performance of the test_dataset
    test_metrics, conf_matrix = evaluate(best_model, test_set, criterion, device)
    # Compute performance metrics
    print_evaluation(test_metrics)
    # Return performance metrics and confusion matrix
    return conf_matrix
  
# Train the deep learning model
def setup_train_evaluate(config, device, train_dataset, val_dataset, test_dataset):
    """
    This function trains a deep learning model and evaluates its performance.

    :param config: The configuration settings for training the model.
    :type config: object
    :param device: The device on which to train the model (e.g., 'cpu' or 'cuda').
    :type device: str
    :param train_dataset: The dataset used for training the model.
    :type train_dataset: torch.utils.data.Dataset
    :param val_dataset: The dataset used for validating the model during training.
    :type val_dataset: torch.utils.data.Dataset
    :param test_dataset: The dataset used for testing the model after training.
    :type test_dataset: torch.utils.data.Dataset
    """
    
    # ---------------------
    # 1. Load data
    # ---------------------    
    
    # --- Oversampling ---
    
    # Compute class count for each class
    targets = torch.tensor(train_dataset.targets)
    class_counts = torch.bincount(targets)
    # Compute weights for each class [weight is inversely proportional to class_count]
    class_weights = 1. / class_counts.float()
    # Give a sample the weight of its class
    sample_weights = class_weights[train_dataset.targets] * config.data.strength_of_oversampling
    # Create a bilanced sampler
    sampler = torch.utils.data.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    # Loading the train_dataset
    train_dl = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.deep_learning_training.batch_size,
        shuffle=False, # Shuffling is mutually exclusive with sampler
        sampler=sampler
    )
    
    # Loading the validation_dataset
    val_dl = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.deep_learning_training.batch_size,
        shuffle=False # Without shuffling the data
    )
    
    # Loading the test_dataset
    test_dl = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.deep_learning_training.batch_size,
        shuffle=False # Without shuffling the data
    )
    print("---------------------")
    
    # ---------------------
    # 2. Load model
    # ---------------------
    
    # Load ResNet model and specify its configuration through the config variable

    model = ResNet(
        ResidualBlock,
        config.ResNet_model.layers,
        config.classification.number_of_classes,
        config.ResNet_model.stride,
        config.ResNet_model.padding,
        config.ResNet_model.kernel,
        config.ResNet_model.channels_of_color,
        config.ResNet_model.planes,
        config.ResNet_model.in_features,
        config.ResNet_model.inplanes
    )
    model.to(device)

    # ---------------------
    # 3. Train model
    # ---------------------
    
    # Set up loss function
    criterion = nn.CrossEntropyLoss()
    
    print("Training ResNet model:\n")
    # Train the model
    best_val_metric, best_model, training_metrics, validation_metrics = train_model(model, config, train_dl, val_dl, device, criterion)
    # Depending on the configuration you choose, create graphs for performance
    if config.graph.create_model_graph:
        print_metrics_graph(training_metrics, validation_metrics, config.graph.metric_plotted_during_traininig, config.graph.view_model_graph)
        
    # --------------------------------
    # 4. Evaluate model on test set
    # --------------------------------
    
    # Evaluate the performance of the model on the test_dataset
    conf_matrix = evaluate_model(best_val_metric, best_model, test_dl, criterion, device)
    print()

    # Print the confusion matrix of the model
    print_confusion_matrix(conf_matrix, config.classification.class_names)
    
    # Depending on the configuration you choose, create graphs for confusion matrix
    if config.graph.create_model_graph:
        print_confusion_matrix_graph(conf_matrix, config.graph.view_model_graph, test=False)
    
    # ---------------------
    # 5. Save model
    # ---------------------
        
    # Verify that the checkpoint folder passed by the configuration parameter exists
    os.makedirs(config.training.checkpoint_dir, exist_ok=True)
    # Store model weights in the checkpoint folder
    torch.save(best_model.state_dict(), f"{config.training.checkpoint_dir}/ResNet_best_model.pt")
    print("---------------------")
    print("ResNet model saved.")


# Main
if __name__ == '__main__':
    """
    The main script to train and evaluate ResNet model.

    The script performs the following steps:
    
    1. Parse configuration
    2. Device configuration
    3. Load data
    4. Train model
    """
    
    # ---------------------
    # 1. Parse configuration
    # ---------------------
    
    # Configuration parameters
    config = Dict(add_arguments())
    
    # ---------------------
    # 2. Device configuration
    # ---------------------
    
    # Selecting the device to run with: CUDA -> GPU; CPU -> CPU
    if config.training.device.lower() == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    # Print selected device
    print("\nDevice: " + torch.cuda.get_device_name()) 
    print("---------------------")
    
    # ---------------------
    # 3. Load data
    # ---------------------
    
    # Create the dataset items splitted in train, validation and test sets
    train_dataset, val_dataset, test_dataset = load_datasets(config)
        
    # ---------------------
    # 4. Train model
    # ---------------------
           
    # Train ResNet model
    setup_train_evaluate(config, device, train_dataset, val_dataset, test_dataset)

    print("---------------------")
    print("\nTrain finish correctly.\n")
