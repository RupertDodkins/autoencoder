import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(784, 128)
        self.encoder_output_layer = nn.Linear(128, 128)

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = F.relu(activation)
        code = self.encoder_output_layer(activation)
        code = F.relu(code)
        return code


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder_hidden_layer = nn.Linear(128, 128)
        self.decoder_output_layer = nn.Linear(128, 784)

    def forward(self, features):
        activation = self.decoder_hidden_layer(features)
        activation = F.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = F.relu(activation)
        return reconstructed


class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, features):
        code = self.encoder(features)
        reconstructed = self.decoder(code)
        return reconstructed