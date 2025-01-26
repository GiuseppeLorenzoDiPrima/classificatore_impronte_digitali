# Third-party imports
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

# Local application/library specific import
from utils import save_graph


# Transformation to apply to the dataset divided according to training, validation and testing
transformation = {
    'training' : transforms.Compose([
            transforms.Resize((224, 224)),  
            transforms.CenterCrop(224),
            transforms.Grayscale(num_output_channels=1),
            transforms.ColorJitter(brightness=0.7, contrast=0.6),
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5))
        ]),
    'validation' : transforms.Compose([
            transforms.Resize((224, 224)),  
            transforms.CenterCrop(224),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5))
        ]),
    'testing' : transforms.Compose([
            transforms.Resize((224, 224)),  
            transforms.CenterCrop(224),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5))
        ]),
}


# Custom class to create PolyU_HRF_DBII dataset
class PolyU_HRF_DBII(Dataset):
    """
    A Dataset for PolyU_HRF_DBII images.
    """
    def __init__(self, type=None, root='data'):
        """
        A Dataset for PolyU_HRF_DBII images.

        :param type: The type of the dataset (e.g. 'train', 'val', 'test').
        :type type: str
        :param root: The root directory of the dataset.
        :type root: str
        """

        # Initialize the variable
        path = ""

        # Initialize the path
        if type == 'train':
            path += str(root.data_dir) + "train//"
        elif type == 'val':
            path += str(root.data_dir) + "val//"
        else:
            path += str(root.data_dir) + "test//"

        # Retrieve data through the ImageFolder class
        if type == 'train':
            self.data = ImageFolder(path, transform=transformation['training'])
        elif type == 'val':
            self.data = ImageFolder(path, transform=transformation['validation'])
        else:
            self.data = ImageFolder(path, transform=transformation['testing'])
        
        # Set variables
        self.classes = self.data.classes
        self.targets = self.data.targets
        self.path = path

    # Return the size of the dataset
    def __len__(self):
        return len(self.data)

    # Return image and label elements via the item dictionary 
    def __getitem__(self, idx):
        if self.data:
            image, label = self.data[idx]
            item = {
                'image' : image,
                'label' : label,
            }
            return item
        else:
            return None


# Count how many items are contained for each class in the dataset
def class_count(dataset):
    """
    Count the number of instances of each class in a dataset.

    :param dataset: The dataset.
    :type dataset: Dataset
    :return: Returns a numpy array containing the counts of each class.
    :rtype: numpy.ndarray
    """

    # Initialize an array of the same size as the classes to zero
    elemets = np.zeros(len(dataset.classes))
    # Retrieve the labels
    labels = dataset.targets
    # Increase the counter for the corresponding class by one
    for value in labels:
        elemets[value] += 1
    # Return a vector containing the number of elements for each class
    return elemets
    
# Print the size of the dataset for each class
def print_shapes(train_dataset, val_dataset, test_dataset):
    """
    Print the shapes of the datasets.

    :param train_dataset: The training dataset.
    :type train_dataset: Dataset
    :param val_dataset: The validation dataset.
    :type val_dataset: Dataset
    :param test_dataset: The test dataset.
    :type test_dataset: Dataset
    """

    # Count classes for each dataset
    train_class_counts = class_count(train_dataset)
    val_class_counts = class_count(val_dataset)
    test_class_counts = class_count(test_dataset)
    
    print("Datasets shapes:\n")

    # Print each dataset class size

    # Train set
    print(f"Train size: {len(train_dataset)}")
    print(f"\t- Train accidental whorl: {int(train_class_counts[0])}")
    print(f"\t- Train central pocket loop whorl: {int(train_class_counts[1])}")
    print(f"\t- Train double loop whorl: {int(train_class_counts[2])}")
    print(f"\t- Train plain arch: {int(train_class_counts[3])}")
    print(f"\t- Train plain whorl: {int(train_class_counts[4])}")
    print(f"\t- Train radial loop: {int(train_class_counts[5])}")
    print(f"\t- Train tended arch: {int(train_class_counts[6])}")
    print(f"\t- Train ulnar loop: {int(train_class_counts[7])}")

    # Validation set
    print(f"Validation size: {len(val_dataset)}")
    print(f"\t- Validation accidental whorl: {int(val_class_counts[0])}")
    print(f"\t- Validation central pocket loop whorl: {int(val_class_counts[1])}")
    print(f"\t- Validation double loop whorl: {int(val_class_counts[2])}")
    print(f"\t- Validation plain arch: {int(val_class_counts[3])}")
    print(f"\t- Validation plain whorl: {int(val_class_counts[4])}")
    print(f"\t- Validation radial loop: {int(val_class_counts[5])}")
    print(f"\t- Validation tended arch: {int(val_class_counts[6])}")
    print(f"\t- Validation ulnar loop: {int(val_class_counts[7])}")

    # Test set
    print(f"Test size: {len(test_dataset)}")
    print(f"\t- Test accidental whorl: {int(test_class_counts[0])}")
    print(f"\t- Test central pocket loop whorl: {int(test_class_counts[1])}")
    print(f"\t- Test double loop whorl: {int(test_class_counts[2])}")
    print(f"\t- Test plain arch: {int(test_class_counts[3])}")
    print(f"\t- Test plain whorl: {int(test_class_counts[4])}")
    print(f"\t- Test radial loop: {int(test_class_counts[5])}")
    print(f"\t- Test tended arch: {int(test_class_counts[6])}")
    print(f"\t- Test ulnar loop: {int(test_class_counts[7])}")

# Print a graph to illustrate the distribution of data across train, validation and test datasets
def visualize_class_distribution(dataset, dataset_name, view):
    """
    Visualize the class distribution in a dataset.

    :param dataset: The dataset to visualize.
    :type dataset: Dataset
    :param dataset_name: The name of the dataset.
    :type dataset_name: str
    :param view: Whether to display the plot.
    :type view: bool
    """

    # Initialize a vector to zero
    class_counts = np.zeros(len(dataset.classes))
    # Fill the vector with the number of elements for each class
    class_counts = class_count(dataset)

    # Print a bar graph according to the colors shown in the order
    plt.bar(dataset.classes, class_counts, color=['blue', 'orange', 'red', 'yellow', 'purple', 'green', 'gray', 'pink'])
    # Add labels and title
    plt.xlabel("Class")
    plt.ylabel("Number of images")
    plt.title(f"Class distribution in the {dataset_name.lower()} dataset")
    # Save graphs in the graph folder
    save_graph(dataset_name, 'Dataset')
    
    # If the user expressed the preference in the base_config file, it shows the result
    if view:
        plt.show()
    # Close the graph to avoid overlap
    plt.close()
    
# Invoke the visualize_class_distribution once for each dataset [Train, Validation and Test]
def print_dataset_graph(train_dataset, val_dataset, test_dataset, view):
    """
    Print the class distribution graph for the train, validation, and test datasets.

    :param train_dataset: The training dataset.
    :type train_dataset: Dataset
    :param val_dataset: The validation dataset.
    :type val_dataset: Dataset
    :param test_dataset: The test dataset.
    :type test_dataset: Dataset
    :param view: Whether to display the plots.
    :type view: bool
    """

    # Print graphs
    print("\nDrawing graph for class distribution in dataset...")
    visualize_class_distribution(train_dataset, "Train", view)
    visualize_class_distribution(val_dataset, "Validation", view)
    visualize_class_distribution(test_dataset, "Test", view)

# Create dataset objects
def load_datasets(config):
    """
    Load the train, validation and test datasets.

    :param config: The configuration for loading the datasets.
    :type config: Config
    :return: Returns the train, validation and test datasets.
    :rtype: tuple (Dataset, Dataset, Dataset)
    """

    # Create objects of the PolyU_HRF_DBII class for each dataset
    train_dataset = PolyU_HRF_DBII(type='train', root=config.data)
    val_dataset  = PolyU_HRF_DBII(type='val', root=config.data)
    test_dataset  = PolyU_HRF_DBII(type='test', root=config.data)
    
    # Print statistics
    print_shapes(train_dataset, val_dataset, test_dataset)

    # If the user expressed the preference in the base_config file, it create the graph
    if config.graph.create_dataset_graph:
        print_dataset_graph(train_dataset, val_dataset, test_dataset, config.graph.view_dataset_graph)

    return train_dataset, val_dataset, test_dataset
