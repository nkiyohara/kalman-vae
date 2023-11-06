from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicsParameterNetwork(ABC):
    @abstractmethod
    def clear_hidden_state(self):
        pass

    @abstractmethod
    def reset_hidden_state(self, batch_size, device):
        pass


class LSTMModel(nn.Module, DynamicsParameterNetwork):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=False
        )
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.hidden = None

    def clear_hidden_state(self):
        self.hidden = None

    def reset_hidden_state(self, batch_size, device, dtype):
        self.hidden = (
            torch.zeros(
                self.num_layers, batch_size, self.hidden_dim, dtype=dtype, device=device
            ),
            torch.zeros(
                self.num_layers, batch_size, self.hidden_dim, dtype=dtype, device=device
            ),
        )

    def forward(self, x):
        if self.hidden is None:
            batch_size = x.size(1)
            self.reset_hidden_state(batch_size, device=x.device, dtype=x.dtype)

        x, self.hidden = self.lstm(x, self.hidden)

        x = self.linear(x)
        x = F.softmax(x, dim=-1)

        return x
