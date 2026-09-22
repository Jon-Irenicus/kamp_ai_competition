import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):

    def __init__(self, input_size, latent_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(self.input_size, self.hidden_size[0])
        self.fc2 = nn.Linear(self.hidden_size[0], self.hidden_size[1])
        self.fc3 = nn.Linear(self.hidden_size[1], self.latent_size)

        self.fc4 = nn.Linear(self.latent_size, self.hidden_size[1])
        self.fc5 = nn.Linear(self.hidden_size[1], self.hidden_size[0])
        self.fc6 = nn.Linear(self.hidden_size[0], self.input_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return self.fc6(x)