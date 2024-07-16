# Parts of this code are inspired by
# GeeksforGeeks (2022): Implementing an Autoencoder in PyTorch. Available online at https://www.geeksforgeeks.org/implementing-an-autoencoder-in-pytorch/, updated on 7/7/2022, checked on 7/16/2024.

import torch

# Creating a PyTorch class
# 28*28 ==> 9 ==> 28*28
class Autoencoder(torch.nn.Module):
    def __init__(self, input_dim, latent_feature_count):
        super().__init__()
        # Building an linear encoder with Linear
        # layer followed by Relu activation function
        # 784 ==> 9
        self.encoder = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(input_dim[0]*input_dim[1], 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, latent_feature_count),
        )

        # Building an linear decoder with Linear
        # layer followed by Relu activation function
        # The Sigmoid activation function
        # outputs the value between 0 and 1
        # 9 ==> 784
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_feature_count, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, input_dim[0]*input_dim[1]),
            torch.nn.Sigmoid(),
            torch.nn.Unflatten(dim=1, unflattened_size=input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
