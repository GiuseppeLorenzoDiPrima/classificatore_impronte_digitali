# Third-party imports
import torch.nn as nn

# Class to define the CNN model by inheriting nn.Module
class CNN(nn.Module):
    """
    A PyTorch implementation of the CNN model.
    """
    def __init__(self, number_of_classes, stride_size, padding_size, kernel_size, channels_of_color, inplace):
        """
        A PyTorch implementation of the CNN model.

        :param type_net: Number of classes to classify
        :type type_net: int
        :param stride_size: The stride size for the convolutional layers.
        :type stride_size: list
        :param padding_size: The padding size for the convolutional layers.
        :type padding_size: list
        :param kernel_size: The kernel size for the convolutional and pooling layers.
        :type kernel_size: list
        :param channels_of_color: The number of color channels in the input images.
        :type channels_of_color: int
        :param inplace: Whether to use inplace ReLU.
        :type inplace: bool
        """
        super(CNN, self).__init__()
        # Conv2d -> Convolutional layer
        # ReLU -> Activation function
        # MaxPool2d -> Pooling layer
        # BatchNorm2d -> Regularization function
        self.features = nn.Sequential(
            nn.Conv2d(channels_of_color, 128, kernel_size=kernel_size[0], stride=stride_size[0], padding=padding_size[0]),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=inplace),
            nn.MaxPool2d(kernel_size=kernel_size[2], stride=stride_size[1]),
            nn.Conv2d(128, 128, kernel_size=kernel_size[1], padding=padding_size[0]),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=inplace),
            nn.Conv2d(128, 256, kernel_size=kernel_size[1], padding=padding_size[0]),
            nn.MaxPool2d(kernel_size=kernel_size[2], stride=stride_size[1]),
            nn.Conv2d(256, 256, kernel_size=kernel_size[2], padding=padding_size[1]),
            nn.ReLU(inplace=inplace),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=kernel_size[2], padding=padding_size[1]),
            nn.ReLU(inplace=inplace),
            nn.Conv2d(512, 512, kernel_size=kernel_size[2], padding=padding_size[1]),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=inplace),
            nn.Conv2d(512, 512, kernel_size=kernel_size[2], padding=padding_size[1]),
            nn.ReLU(inplace=inplace),
            nn.Conv2d(512, 512, kernel_size=kernel_size[2], padding=padding_size[1]),
            nn.ReLU(inplace=inplace),
            nn.MaxPool2d(kernel_size=kernel_size[2], stride=stride_size[1]),
        )
        # Dropout -> Regularization function
        # Linear -> Linear layer
        # ReLU -> Activation function
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 6 * 6, 4096),
            nn.ReLU(inplace=inplace),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=inplace),
            nn.Linear(4096, number_of_classes),
        )
    
    # Foward step
    def forward(self, x):
        """
        Defines the forward pass of the CNN.

        :param x: The input to the network.
        :type x: torch.Tensor
        :return: Returns the output of the network.
        :rtype: torch.Tensor
        """
        x = self.features(x)
        x = x.view(x.size(0), 512 * 6 * 6)
        x = self.classifier(x)
        return x
