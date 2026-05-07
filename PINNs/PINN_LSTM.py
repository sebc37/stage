import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np


class FourierEmbedding(nn.Module):
    def __init__(self, in_dim, embed_dim):
        super().__init__()
        self.B = nn.Parameter(torch.randn(in_dim, embed_dim) * 10, requires_grad=False)
        

    def forward(self, x):
        # x: (batch, seq_len, in_dim)
        x_proj = 2 * torch.pi * x @ self.B  # (batch, seq_len, embed_dim)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    

class PINN(nn.Module):
    def __init__(self, input_dim, ic_len=10, hidden_dim=64, lstm_hidden=128, fcn_dim=512, output_dim=3):
        super().__init__()
        
        self.ic_len = ic_len
        self.lstm_hidden = lstm_hidden

        # Fourier embedding
        self.embedding = FourierEmbedding(input_dim, hidden_dim)
        embedded_dim = hidden_dim * 2

        # LSTM
        self.lstm = nn.LSTM(
            input_size=embedded_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True
        )

        # Encodeur des conditions initiales → h0 et c0
        self.ic_encoder = nn.Linear(ic_len * 3, 2 * lstm_hidden)

        # FCN
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, fcn_dim),
            nn.ReLU(),
            nn.Linear(fcn_dim, fcn_dim),
            nn.ReLU(),
            nn.Linear(fcn_dim, output_dim)
        )

    def forward(self, x, ic):
        """
        x: (batch, seq_len, input_dim)
        ic: (batch, ic_len, 3)
        """

        batch_size = x.size(0)

        # -------- Encodage des conditions initiales --------
        ic_flat = ic.view(batch_size, -1)  # (batch, ic_len * 3)
        ic_encoded = self.ic_encoder(ic_flat)  # (batch, 2 * lstm_hidden)

        h0, c0 = torch.chunk(ic_encoded, 2, dim=-1)  # chacun (batch, lstm_hidden)

        # reshape pour LSTM: (num_layers=1, batch, hidden)
        h0 = h0.unsqueeze(0)
        c0 = c0.unsqueeze(0)

        # -------- Fourier embedding --------
        x = self.embedding(x)  # (batch, seq_len, embedded_dim)

        # -------- LSTM avec états initiaux --------
        lstm_out, _ = self.lstm(x, (h0, c0))  # (batch, seq_len, lstm_hidden)

        # dernière sortie temporelle
        last_output = lstm_out[:, -1, :]  # (batch, lstm_hidden)

        # -------- FCN --------
        out = self.fc(last_output)  # (batch, 3)

        return out


class TrajectoryDataset(Dataset):
    def __init__(self, data, seq_len, ic_len=10, dt=1.0, random_sampling=True):
        """
        data: numpy array (N, 3) -> x, y, z
        seq_len: longueur totale de la séquence (n)
        ic_len: nombre de conditions initiales (10)
        dt: pas de temps
        random_sampling: tirage aléatoire ou séquentiel
        """
        assert data.shape[1] == 3, "Les données doivent être (N, 3)"
        assert seq_len > ic_len, "seq_len doit être > ic_len"

        self.data = data
        self.seq_len = seq_len
        self.ic_len = ic_len
        self.dt = dt
        self.random_sampling = random_sampling

        self.max_start = len(data) - seq_len

    def __len__(self):
        return len(self.data) if self.random_sampling else self.max_start

    def __getitem__(self, idx):
        if self.random_sampling:
            start = np.random.randint(0, self.max_start)
        else:
            start = idx

        seq = self.data[start:start + self.seq_len]  # (seq_len, 3)

        # temps relatif à t0
        t = np.arange(self.seq_len) * self.dt  # (seq_len,)

        # conditions initiales (10 premiers points)
        ic = seq[:self.ic_len]  # (10, 3)

        # construction de l'entrée :
        # concat temps + état (x,y,z)
        t = t[:, None]  # (seq_len, 1)
        x_input = np.concatenate([t, seq], axis=1)  # (seq_len, 4)

        # target = dernier point (x,y,z)
        y = seq[-1]

        return (
            torch.tensor(x_input, dtype=torch.float32),  # (seq_len, 4)
            torch.tensor(ic, dtype=torch.float32),       # (10, 3)
            torch.tensor(y, dtype=torch.float32)         # (3,)
        )