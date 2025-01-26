#-----  Command to run from terminal  -----#
# python -u test.py -c config/base_config.yaml

# Third-party imports
import torch
import torch.nn as nn

# Local application/library specific imports
from data_classes.manage_dataset import PolyU_HRF_DBII
from model_classes.resnet_model import ResNet, ResidualBlock
from utils import *

# Configuration and utility imports
from yaml_config_override import add_arguments
from addict import Dict

# Print test set performance metrics
def print_metrics(metrics):
    """
    Prints the metrics.

    :param metrics: Dictionary of metrics to print.
    :type metrics: Dict
    """

    print("\nResNet model performance:\n")
    # Scrolls through the dictionary and prints performance metrics
    for key, value in metrics.items():
        print(f"Test {key}: {value:.4f}")

# Test the ResNet model
def test_model(config, device, test_dataset):
    """
    This function tests a deep learning model on a test dataset.

    :param config: The configuration settings for testing the model.
    :type config: object
    :param device: The device on which to test the model (e.g., 'cpu', 'cuda').
    :type device: str
    :param test_dataset: The dataset used for testing the model.
    :type test_dataset: torch.utils.data.Dataset
    :return: Returns the metrics for the test dataset.
    :rtype: dict
    """
    
    # ---------------------
    # 1. Load data
    # ---------------------
    
    # Loading the test_dataset
    test_dl = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.deep_learning_training.batch_size,
        shuffle=False # Without shuffling the data
    )
    
    # ---------------------
    # 2. Load model
    # ---------------------
    
    # Load ResNet Model and specify its configuration through the config variable
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
    # 3. Load model weights
    # ---------------------
    
    # Loads the saved model weights to the specified folder during training
    print("Loading ResNet model...")
    model.load_state_dict(torch.load(f"{config.training.checkpoint_dir}/ResNet_best_model.pt"))
    print("-> ResNet model loaded.")
    print("---------------------")
    
    # ---------------------
    # 4. Criterion
    # ---------------------
    
    # Defines the CrossEntropyLoss as loss functions for ResNet model
    criterion = nn.CrossEntropyLoss()
    
    # ---------------------
    # 5. Evaluate
    # ---------------------
    
    print("Evaluating model...\n")
    # Evaluate model performance
    metrics, conf_matrix = evaluate(model, test_dl, criterion, device)
    # Prints the confusion matrix of the model
    print_confusion_matrix(conf_matrix, config.classification.class_names)
    print("---------------------")
    # Print confusion matrices graphs
    if config.graph.create_model_graph:
        print_confusion_matrix_graph(conf_matrix, config.graph.view_model_graph, test=True)

    # ---------------------
    # 6. Print performance
    # ---------------------
    
    print("Performance:")
    print_metrics(metrics)
    if config.graph.create_model_graph:
        print_test_metrics_graph(metrics, config.graph.metric_plotted_during_testing, config.graph.view_model_graph)


# Main
if __name__ == '__main__':
    """
    The main script to test the ResNet model.

    The script performs the following steps:
    
    1. Load configuration
    2. Set device
    3. Load data
    4. Verify the presence of saved model
    5. Test on saved model
    6. Print performance    
    """
    
    # ---------------------
    # 1. Load configuration
    # ---------------------
    
    # Configuration parameters
    config = Dict(add_arguments())
    
    # ---------------------
    # 2. Set device
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
    
    # Create the test_dataset item
    test_dataset  = PolyU_HRF_DBII(type='test', root=config.data)
    
    # ---------------------
    # 4. Verify the presence of saved model
    # ---------------------
    
    path = os.getcwd()
    # No checkpoints directory
    if not os.path.exists(os.path.join(path + "/", config.training.checkpoint_dir)):
        os.makedirs(os.path.join(path + "/", config.training.checkpoint_dir))
        raise Exception("Error no checkpoints directory. It has been created right now.")
    
    # No ResNet_best_model.pt
    if not os.path.isfile(path + "/" + config.training.checkpoint_dir + "/ResNet_best_model.pt"):
        raise Exception("Error no saved model.")
    
    # ---------------------
    # 5. Test on saved models
    # ---------------------
    
    # Test ResNet model
    test_model(config, device, test_dataset)

    print("---------------------")
    print("\nTest finish correctly.\n")
