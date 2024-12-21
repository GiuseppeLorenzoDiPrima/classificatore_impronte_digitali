#-----  Command to run from terminal  -----#
# python -u classify.py -c config/base_config.yaml

# Third-party imports
import torch
import shutil
from joblib import load
from PIL import Image
import os
from torchvision import transforms
from torch.utils.data import Dataset

# Local application/library specific imports
from model_classes.resnet_model import ResNet, ResidualBlock
from model_classes.CNN_model import CNN
from extract_representations.vision_embeddings import VisionEmbeddings
from sklearn import svm

# Configuration and utility imports
from yaml_config_override import add_arguments
from addict import Dict

# Define classes
class_names = ['Accidental Whorl', 'Central Pocket Loop Whorl', 'Double Loop Whorl', 'Plain Arch', 'Plain Whorl', 'Radial Loop', 'Tended Arch', 'Ulnar Loop']

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

# Classify with machine learning model
def classify_ml_model(model_name, config, dataset):
    """
    This function classifies with a machine learning model on the created dataset.

    :param model_name: The name of the machine learning model to use (e.g. 'SVM').
    :type model_name: str
    :param config: The configuration settings for the model.
    :type config: object
    :param dataset: The dataset used for classification.
    :type dataset: torch.utils.data.Dataset
    """
    
    # ---------------------
    # 1. Compute dataset for svm
    # ---------------------
    
    print("Vision embeddings for SVM:\n")
    # Load the pca object determined during the training phase
    pca = load(f"{config.training.checkpoint_dir}/pca.joblib")
    # Create vision_embedding object
    vision_embeddings = VisionEmbeddings()
    # Create the dataset containing features for the svm model
    dataset_svm = vision_embeddings.extract_single_dataset(dataset, pca, 'classification', False, False)
    print("---------------------")
    
    # ---------------------
    # 2. Load model
    # ---------------------
    
    # Load the templates and specify their configuration through the config variable
    # SVM model
    svm_model = svm.SVC(
        gamma=config.svm_training.gamma,
        kernel=config.svm_training.kernel,
        C=config.svm_training.C,
        probability=config.svm_training.probability
    )
    
    # ---------------------
    # 3. Load model weights
    # ---------------------
    
    # Loads the saved model weights to the specified folder during training
    print("Loading " + model_name + " model...")
    # SVM model
    svm_model = load(f"{config.training.checkpoint_dir}/{model_name}_best_model.pkl")
    print("-> " + model_name + " model loaded.")
    print("---------------------")
    print("Using " + model_name + " model for classification:")
    
    # ---------------------
    # 4. Classify
    # ---------------------
    
    # Classify and move image to corresponding sub-folder
    for features, img_paths in zip(dataset_svm.features, dataset_svm.path):
        # Compute corresponding class
        pred = svm_model.predict([features])[0]
        class_name = class_names[pred]
        
        # Select/cresate sub-folder
        class_folder = os.path.join(config.classification.output_folder, class_name)
        os.makedirs(class_folder, exist_ok=True)
        
        # Move image to corrisponding folder
        shutil.move(img_paths, os.path.join(class_folder, os.path.basename(img_paths)))
        
        # Print result
        print(f"Image {os.path.basename(img_paths)} classified as: {class_name}.")

# Classify with deep learning model
def classify_dl_model(model_name, config, device, dataset):
    """
    This function classifies with a deep learning model on the created dataset.

    :param model_name: The name of the deep learning model to use (e.g. 'ResNet' or 'CNN').
    :type model_name: str
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
    
    # Load the templates and specify their configuration through the config variable
    if 'resnet' in model_name.lower():
        # ResNet Model
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
    
    # CNN Model
    else:
        model = CNN(
            config.classification.number_of_classes,
            config.CNN_model.stride,
            config.CNN_model.padding,
            config.CNN_model.kernel,
            config.CNN_model.channels_of_color,
            config.CNN_model.inplace,
        )
        model.to(device)

    # ---------------------
    # 3. Load model weights
    # ---------------------
    
    # Loads the saved model weights to the specified folder during training
    print("Loading " + model_name + " model...")
    model.load_state_dict(torch.load(f"{config.training.checkpoint_dir}/{model_name}_best_model.pt"))
    print("-> " + model_name + " model loaded.")
    print("---------------------")
    print("Using " + model_name + " model for classification:")
    
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
                class_name = class_names[pred]
                class_folder = os.path.join(config.classification.output_folder, class_name)
                os.makedirs(class_folder, exist_ok=True)

                # Move image to corrisponding folder
                shutil.move(img_path, os.path.join(class_folder, os.path.basename(img_path)))

                # Print result
                print(f"Image {os.path.basename(img_path)} classified as: {class_name}.")


# Main
if __name__ == '__main__':
    """
    The main script for classifing fingerprints.

    The script performs the following steps:
    
    1. Load configuration
    2. Set device
    3. Load data
    4. Get saved model name
    5. Classify on saved models and move classified fingerpints  
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
        raise Exception("Error no \"Fingerprint_to_classify\" found. It has been created right now. Please insert image you want to classify.")
    if not os.path.exists(os.path.join(path + "/", config.classification.output_folder)):
        os.makedirs(os.path.join(path + "/", config.classification.output_folder))

    # Load fingerprints
    dataset = PolyU_HRF_DBII(config.classification.image_folder, transform)

    # ---------------------
    # 4. Get saved model name
    # ---------------------
    
    if not os.path.exists(os.path.join(path + "/", config.training.checkpoint_dir)):
        os.makedirs(os.path.join(path + "/", config.training.checkpoint_dir))
    # Get path of saved models
    saved_models_path = os.listdir(os.path.join(path + "/", config.training.checkpoint_dir))
    saved_models = [path.lower() for path in saved_models_path]

    # ---------------------
    # 5. Classify on saved models
    # ---------------------

    # Based on performance: ResNet >> CNN >> SVM

    # ResNet
    if 'resnet_best_model.pt' in saved_models:
        model = torch.load(config.training.checkpoint_dir + "ResNet_best_model.pt")
        classify_dl_model("ResNet", config, device, dataset)
    
    # CNN
    elif 'cnn_best_model.pt' in saved_models:
        model = torch.load(config.training.checkpoint_dir + "CNN_best_model.pt")
        classify_dl_model("CNN", config, device, dataset)
    
    # SVM
    elif 'svm_best_model.pkl' in saved_models:
        model = load(config.training.checkpoint_dir + "SVM_best_model.pkl")
        classify_ml_model("SVM", config, dataset)
    
    # No model saved
    else:
        raise Exception("Error no model saved.")

    print("\nClassification finish correctly.\n")
