import torch


# Neural Network
class Cnn(torch.nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        # Input channels = 1, output channels = 6
        self.conv1 = torch.nn.Conv1d(1, 6, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool1d(kernel_size=2)

        self.conv2 = torch.nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # 4608 input features, 64 output features (see sizing flow below)
        self.fc1 = torch.nn.Linear(16 * 4 * 4, 32)

        # 64 input features, 10 output features for our 10 defined classes
        self.fc2 = torch.nn.Linear(32, 10)

    def forward(self, x):
        # Computes the activation of the first convolution
        # Size changes from (1, 16, 16) to (6, 16, 16)
        x = torch.nn.functional.relu(self.conv1(x))

        # Size changes from (6, 16, 16) to (6, 8, 8)
        x = self.pool(x)

        # Size changes from (6, 8, 8) to (16, 8, 8)
        x = torch.nn.functional.relu(self.conv2(x))

        # Size changes from (16, 8, 8) to (16, 4, 4)
        x = self.pool2(x)

        # Reshape data to input to the input layer of the neural net
        # Size changes from (16, 4, 4) to (1, 256)
        # Recall that the -1 infers this dimension from the other given dimension
        x = x.view(-1, 16 * 4 * 4)

        # Computes the activation of the first fully connected layer
        # Size changes from (1, 384) to (1, 32)
        x = torch.nn.functional.relu(self.fc1(x))

        # Computes the second fully connected layer (activation applied later)
        # Size changes from (1, 32) to (1, 10)
        x = self.fc2(x)
        return x
