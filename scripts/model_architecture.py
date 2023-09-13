# Define the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3, 24)
        self.bn1 = nn.BatchNorm1d(24)  # Batch normalization after the first hidden layer
        self.fc2 = nn.Linear(24, 24)
        self.bn2 = nn.BatchNorm1d(24)  # Batch normalization after the second hidden layer
        #self.fc3 = nn.Linear(100, 100)
        #self.bn3 = nn.BatchNorm1d(100)  # Batch normalization after the third hidden layer
        self.fc4 = nn.Linear(24, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.bn1(x)  # Apply batch normalization
        x = torch.tanh(self.fc2(x))
        x = self.bn2(x)  # Apply batch normalization
        #x = F.leaky_relu(self.fc3(x))
        #x = self.bn3(x)  # Apply batch normalization
        x = self.fc4(x)
        return x