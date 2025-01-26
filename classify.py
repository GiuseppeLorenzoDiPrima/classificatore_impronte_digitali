#-----  Command to run from terminal  -----#
# python -u classify.py -c config/base_config.yaml

# Third-party imports
import torch
import shutil
from PIL import Image
import os
from torchvision import transforms
from torch.utils.data import Dataset

# Local application/library specific imports
from model_classes.resnet_model import ResNet, ResidualBlock

# Configuration and utility imports
from yaml_config_override import add_arguments
from addict import Dict

# Transformation to apply to the dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5))
])


# Custom class to create PolyU_HRF_DBII dataset
class PolyU_HRF_DBII(Dataset):
    """
    A Dataset for PolyU_HRF_DBII images.
    """

    def __init__(self, image_folder, transform=None):
        """
        A Dataset for PolyU_HRF_DBII images.

        :param image_folder: Folder with images to classify.
        :type image_folder: str
        :param transform: Transformation to apply to the dataset
        :type transform: torchvision.transform
        """

        self.image_folder = image_folder # Set folder with images to classify
        self.transform = transform # Set transormation
        self.classes = None
        self.targets = None
        self.path = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))] # Select only image

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        img_path = self.path[idx] # Compute image path
        image = Image.open(img_path).convert('RGB') # Load image
        if self.transform: # Apply tranformation
            image = self.transform(image)
        item = {
                'image' : image,
                'image_path' : img_path
            }
        return item


# Classify with ResNet model
def classify(config, device, dataset):
    """
    This function classifies with a ResNet model on the created dataset.

    :param config: The configuration settings for the model.
    :type config: object
    :param device: The device on which to compute classification (e.g. 'cpu' or 'cuda').
    :type device: str
    :param dataset: The dataset used for classification.
    :type test_dataset: torch.utils.data.Dataset
    """
    
    # ---------------------
    # 1. Load data
    # ---------------------
    
    # Create a dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.deep_learning_training.batch_size,
        shuffle=False # Without shuffling the data
    )
    
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
    # 3. Load model weights
    # ---------------------
    
    # Loads the saved model weights to the specified folder during training
    print("Loading ResNet model...")
    model.load_state_dict(torch.load(f"{config.training.checkpoint_dir}/ResNet_best_model.pt"))
    print("-> ResNet model loaded.")
    print("---------------------")
    print("Using ResNet model for classification:")
    
    # ---------------------
    # 4. Classify
    # ---------------------
    
    # Set model to evaluation mode
    model.eval()

    # Do not compute gradient
    with torch.no_grad():
        for batch in dataloader:
            # Get images and targets
            images = batch['image'].to(device)
            img_paths = batch['image_path']
            # Load image on device (cpu or gpu)
            images = images.to(device)
            # Compute corresponding class
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            # Select/cresate sub-folder
            for img_path, pred in zip(img_paths, preds):
                class_name = config.classification.class_names[pred]
                class_folder = os.path.join(config.classification.output_folder, class_name)
                os.makedirs(class_folder, exist_ok=True)

                # Move image to corrisponding folder
                shutil.move(img_path, os.path.join(class_folder, os.path.basename(img_path)))

                # Print result
                print(f"Image {os.path.basename(img_path)} classified as: {class_name}.")


# Main
if __name__ == '__main__':
    """
    The main script to classify fingerprints.

    The script performs the following steps:
    
    1. Load configuration
    2. Set device
    3. Load data
    4. Verify the presence of saved model
    5. Classify on saved model and move classified fingerpints  
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
    
    # Set/create folder for classification
    path = os.getcwd()
    if not os.path.exists(os.path.join(path + "/", config.classification.image_folder)):
        os.makedirs(os.path.join(path + "/", config.classification.image_folder))
        raise Exception("Error no \"Fingerprint_to_classify\" directory found. It has been created right now. Please insert there image you want to classify.")
    if not os.path.exists(os.path.join(path + "/", config.classification.output_folder)):
        os.makedirs(os.path.join(path + "/", config.classification.output_folder))

    # Load fingerprints
    dataset = PolyU_HRF_DBII(config.classification.image_folder, transform)

    # ---------------------
    # 4. Verify the presence of saved model
    # ---------------------
    
    # No checkpoints directory
    if not os.path.exists(os.path.join(path + "/", config.training.checkpoint_dir)):
        os.makedirs(os.path.join(path + "/", config.training.checkpoint_dir))
        raise Exception("Error no checkpoints directory. It has been created right now.")
    
    # No ResNet_best_model.pt
    if not os.path.isfile(path + "/" + config.training.checkpoint_dir + "/ResNet_best_model.pt"):
        raise Exception("Error no saved model.")

    # ---------------------
    # 5. Classify on saved model
    # ---------------------

    # Classify with ResNet model
    classify(config, device, dataset)

    print("\nClassification finish correctly.\n")
