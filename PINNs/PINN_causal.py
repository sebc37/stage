import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

# =========================
# MLP
# =========================
class MLP(nn.Module):
    def __init__(self, layers):
        super().__init__()
        net = []
        for i in range(len(layers)-2):
            net.append(nn.Linear(layers[i], layers[i+1]))
            net.append(nn.Tanh())
        net.append(nn.Linear(layers[-2], layers[-1]))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


# =========================
# PINN
# =========================
class PINN:
    def __init__(self, layers, states0, t0, t1, tol, device="cpu"):

        self.device = device

        self.states0 = torch.tensor(states0, dtype=torch.float32).to(device)
        self.t0 = t0
        self.t1 = t1
        self.tol = tol

        # Grid
        n_t = 300
        eps = 0.1 * t1
        self.t = torch.linspace(t0, t1 + eps, n_t).view(-1,1).to(device)

        self.M = torch.triu(torch.ones(n_t, n_t), diagonal=1).T.to(device)

        # Lorenz params
        self.rho = 28.0
        self.sigma = 10.0
        self.beta = 8.0 / 3.0

        # Model
        self.model = MLP(layers).to(device)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        # Logs
        self.loss_log = []

    # =========================
    # Forward PINN
    # =========================
    def neural_net(self, t):
        t.requires_grad_(True)
        out = self.model(t) * t

        x = out[:,0] + self.states0[0]
        y = out[:,1] + self.states0[1]
        z = out[:,2] + self.states0[2]

        return x, y, z, t

    # =========================
    # Residuals
    # =========================
    def residuals(self, t):

        x, y, z, t = self.neural_net(t)

        x_t = torch.autograd.grad(x, t, grad_outputs=torch.ones_like(x), create_graph=True)[0]
        y_t = torch.autograd.grad(y, t, grad_outputs=torch.ones_like(y), create_graph=True)[0]
        z_t = torch.autograd.grad(z, t, grad_outputs=torch.ones_like(z), create_graph=True)[0]

        r1 = x_t.squeeze() - self.sigma * (y - x)
        r2 = y_t.squeeze() - x * (self.rho - z) + y
        r3 = z_t.squeeze() - x * y + self.beta * z

        return r1, r2, r3

    # =========================
    # Loss
    # =========================
    def loss_res(self):
        r1, r2, r3 = self.residuals(self.t)

        R = r1**2 + r2**2 + r3**2

        # causal weights
        W = torch.exp(- self.tol * (self.M @ R.detach()))

        return torch.mean(W * R)

    def loss(self):
        return self.loss_res()

    # =========================
    # Train
    # =========================
    def train(self, nIter=10000):

        pbar = trange(nIter)

        for it in pbar:

            self.optimizer.zero_grad()
            loss = self.loss()
            loss.backward()
            self.optimizer.step()

            if it % 1000 == 0:
                self.loss_log.append(loss.item())

                pbar.set_postfix({"loss": loss.item()})

    # =========================
    # Prediction
    # =========================
    def predict(self, t):
        t = torch.tensor(t, dtype=torch.float32).view(-1,1).to(self.device)
        x, y, z, _ = self.neural_net(t)
        return x.detach().cpu().numpy(), \
               y.detach().cpu().numpy(), \
               z.detach().cpu().numpy()
    
layers = [1, 128, 128, 128, 3]

model = PINN(
    layers=layers,
    states0=[8.0, 0.0, 30.0],
    t0=0.0,
    t1=1.0,
    tol=0.99,
    device="cpu"
)

model.train(10000)

t_test = np.linspace(0, 1000, 1000)
x, y, z = model.predict(t_test)

plt.figure()
plt.plot(t_test, x, label="x pred")
plt.plot(t_test, y, label="y pred")
plt.plot(t_test, z, label="z pred")
plt.legend()
plt.show()