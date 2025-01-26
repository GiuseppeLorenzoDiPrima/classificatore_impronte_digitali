# Third-party imports
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Standard library imports
import os


# Calculate performance metrics
def compute_metrics(predictions, references):
    """
    Computes accuracy, precision, recall, and F1 score.

    :param predictions: The predicted labels.
    :type predictions: List
    :param references: The true labels.
    :type references: List
    :return: A dictionary containing the accuracy, precision, recall, and F1 score.
    :rtype: Dictionary
    """

    # Compute performance metrics: accuracy, precision, recall and f1
    acc = accuracy_score(references, predictions)
    precision = precision_score(references, predictions, average='macro', zero_division=0.0)
    recall = recall_score(references, predictions, average='macro', zero_division=0.0)
    f1 = f1_score(references, predictions, average='macro', zero_division=0.0)
    
    # Return metrics to a dictionary
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Evaluate performance metrics and confusion matrix
def evaluate(model, dataloader, criterion, device):
    """
    Evaluates a model on a given dataset.

    :param model: The model to be evaluated.
    :type model: torch.nn.Module
    :param dataloader: The DataLoader for the dataset.
    :type dataloader: torch.utils.data.DataLoader
    :param criterion: The criterion to use for calculating loss during evaluation.
    :type criterion: torch.nn.modules.loss._Loss
    :param device: The device on which to evaluate the model (e.g., 'cpu', 'cuda').
    :type device: String
    :return: Returns the evaluation metrics and confusion matrix.
    :rtype: tuple (dict, numpy.ndarray)
    """

    # Set the model to evaluation mode
    model.eval()
    # Initialize variables
    running_loss = 0.0
    predictions = []
    references = []

    # Specify that you don't want to calculate the gradient to save computational power
    with torch.no_grad():
        # Iterates through all batches in the dataloader
        for batch in dataloader:
            # Get images and targets
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            # Calculate output
            outputs = model(images)
            # Calculate the loss through the previously chosen loss function
            loss = criterion(outputs, labels)
            # Add the current loss to the total
            running_loss += loss.item()
            # Compute predictions
            pred = torch.argmax(outputs, dim=1)
            predictions.extend(pred.cpu().numpy())
            # Compute refereces
            references.extend(labels.cpu().numpy())

    # Compute performance metrics based on differences between predictiones and references
    val_metrics = compute_metrics(predictions, references)
    # Add loss to performance metrics
    val_metrics['loss'] = running_loss / len(dataloader)
    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(predictions, references)
    # Return metrics and confusion matrix
    return val_metrics, conf_matrix

# Create the graph for train and validation performance metrics
def print_metrics_graph(training_metrics, validation_metrics, metric_plotted, view):
    """
    Prints the graph of a given metric for the training and validation datasets.

    :param training_metrics: The training metrics.
    :type training_metrics: List
    :param validation_metrics: The validation metrics.
    :type validation_metrics: List
    :param metric_plotted: The metric to be plotted.
    :type metric_plotted: String
    :param view: Whether to display the plot.
    :type view: bool
    """

    # Print the graph with for all epochs for training and validation for each performance metric
    
    # For deep learning model
    for element in metric_plotted:
        plt.plot([metrics[element] for metrics in training_metrics], label = 'Training')
        plt.plot([metrics[element] for metrics in validation_metrics], label = 'Validation')
        plt.legend()
        plt.title("Graph of " + str(element) + " per epoch for ResNet model:")
        # Improves graph visibility
        plt.tight_layout()
        save_graph(str('Graph of ' + str(element)), "ResNet model")
        # Check if your configuration likes a print or not
        if view:
            plt.show()
        # Close the graph to avoid overlap
        plt.close()

# Create the graph for test performance metrics
def print_test_metrics_graph(metrics, metric_plotted, view):
    """
    Prints the graph of a given metric for the testing datasets.

    :param metrics_list: The metrics.
    :type metrics_list: List
    :param metric_plotted: The metric to be plotted.
    :type metric_plotted: String
    :param view: Whether to display the plot.
    :type view: bool
    """

    # Print the graph for test for each performance metric
    for element in metric_plotted:
        value = metrics[element]

        plt.figure()
        plt.bar(0, value, width=0.4, label=element.capitalize())
        plt.legend()
        plt.title(f"Graph of {element} for ResNet model")
        plt.yticks([*plt.yticks()[0], value])
        plt.xlim(-0.5, 0.5)
        plt.tight_layout()

        # Save the graph
        save_graph(f'Graph of {element}', "Testing result")
        if view:
            plt.show()
        plt.close()

# Save the created graph
def save_graph(filename, directory):
    """
    Saves a graph to a file.

    :param filename: The name of the file to save the graph to.
    :type filename: String
    :param type_of_graph: The type of the graph (e.g. 'Dataset', 'ResNet model', 'Testing result').
    :type type_of_graph: String
    """

    # Get the current path
    path = os.getcwd()
    # Check if the graph folder exists, if not, create it
    if not os.path.exists(os.path.join(path, 'graph')):
        os.makedirs(os.path.join(path, 'graph'))
    # Check if the type_of_graph subfolder exists, if not, create it
    if not os.path.exists(os.path.join((str(path) + '//graph'), directory)):
        os.makedirs(os.path.join((str(path) + '//graph'), directory))
    # Save the graph
    plt.savefig(str(str(path) + '//graph//' + str(directory) + '//' + str(filename) + '.png'))

# Create a graph for the confusion matrix
def print_confusion_matrix_graph(conf_matrix, view, test):
    """
    Prints a confusion matrix graph for a model.

    :param conf_matrix: The confusion matrix to plot.
    :type conf_matrix: numpy.ndarray
    :param view: Whether to display the plot.
    :type view: bool
    :param test: Whether the model is in the testing phase.
    :type test: Bool
    """

    # Select color
    sns.color_palette("YlOrBr", as_cmap=True)
    # Create a heatmap with confusion matrix
    sns.heatmap(conf_matrix, annot=True)
    # Set labels
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    # Set title
    plt.title(f"Heatmap of confusion matrix for ResNet model")
    # Improves graph visibility
    plt.tight_layout()
    # Save the graph to a specific path
    if test:
        save_graph("ResNet model\'s heatmap", 'Testing result')
    else:
        save_graph('Heatmap of confusion matrix', "ResNet model")
    # Check if your configuration likes a print or not
    if view:
        plt.show()
    # Close the graph to avoid overlap
    plt.close()

# Print confusion matrix on the screen
def print_confusion_matrix(conf_matrix, class_names):
    """
    Prints a confusion matrix for the model.

    :param conf_matrix: The confusion matrix to print.
    :type conf_matrix: Numpy.ndarray
    """

    print("Confusion matrix for ResNet model:")
    # Print the confusion matrix with DataFrame
    df_confusion_matrix = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)
    print(df_confusion_matrix)

# Prints the best evaluation metrics found during the evaluation phase
def print_best_val_metrics(best_val_metric):
    """
    Prints the best model performance on the validation dataset.

    :param best_val_metric: The best validation metrics.
    :type best_val_metric: Dictionary
    """

    print("Best ResNet model performance on validation dataset:")
    # Print the performance of the best model based on the validation_dataset on which the test_dataset will be tested
    for key, value in best_val_metric.items():
        print(f"\t- Best ResNet model {key}: {value:.4f}")
    print("\nTesting ResNet model on test dataset...")

# Print the result of the evaluation
def print_evaluation(test_metrics):
    """
    Prints the evaluation metrics and returns them as a list.

    :param test_metrics: The test metrics.
    :type test_metrics: Dictionary
    :return: The test metrics as a list.
    :rtype: List
    """

    # Store performance in lists so you can pass it to the DataFrame
    metrics = [test_metrics['accuracy'], test_metrics['precision'], test_metrics['recall'], test_metrics['f1'], test_metrics['loss']]
    labels = ['Test accuracy', 'Test precision', 'Test recall', 'Test f1 score', 'Test loss']
    # Print performance on the test_dataset
    test_result = pd.DataFrame(metrics, columns=['Value'], index=labels).round(4)
    print(test_result)
