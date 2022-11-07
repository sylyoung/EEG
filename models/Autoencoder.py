import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, encoder_neurons, decoder_neurons):
        super().__init__()

        encoder_layers = []
        for i in range(len(encoder_neurons) - 1):
            encoder_layers.append(nn.Linear(encoder_neurons[i], encoder_neurons[i+1]))
            if i != len(encoder_neurons) - 2:
                encoder_layers.append(nn.ELU())
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        for i in range(len(decoder_neurons) - 1):
            decoder_layers.append(nn.Linear(decoder_neurons[i], decoder_neurons[i+1]))
            if i != len(decoder_neurons) - 2:
                decoder_layers.append(nn.ELU())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class Autoencoder_encoder(nn.Module):
    def __init__(self, encoder_neurons):
        super().__init__()

        encoder_layers = []
        for i in range(len(encoder_neurons) - 1):
            encoder_layers.append(nn.Linear(encoder_neurons[i], encoder_neurons[i+1]))
            if i != len(encoder_neurons) - 2:
                encoder_layers.append(nn.ELU())
        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded


class Autoencoder_decoder(nn.Module):
    def __init__(self, decoder_neurons):
        super().__init__()

        decoder_layers = []
        for i in range(len(decoder_neurons) - 1):
            decoder_layers.append(nn.Linear(decoder_neurons[i], decoder_neurons[i+1]))
            if i != len(decoder_neurons) - 2:
                decoder_layers.append(nn.ELU())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        decoded = self.decoder(x)
        return decoded