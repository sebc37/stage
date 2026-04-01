import torch
import torch.nn as nn
from torch.func import functional_call, jacrev, vmap

# =========================
# 1. MODELE
# =========================
class MLP(nn.Module):
    def __init__(self, in_dim=1, hidden_dim=64, out_dim=1, depth=3):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers += [nn.Linear(hidden_dim, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# =========================
# 2. PHYSIQUE
# =========================
def f_physics(u):
    return -u  # exemple simple


# =========================
# 3. RESIDU (fonctionnel)
# =========================
def residual_fn(params, model, t):
    t = t.requires_grad_(True)

    u = functional_call(model, params, (t,))

    du_dt = torch.autograd.grad(
        u,
        t,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    R = du_dt - f_physics(u)
    return R.squeeze()  # (N,)


# =========================
# 4. UTILITAIRE
# =========================
def flatten_params(params):
    return torch.cat([p.reshape(-1) for p in params.values()])


# =========================
# 5. NTK OPTIMISÉ
# =========================
def compute_ntk_fast(model, t):
    params = dict(model.named_parameters())

    # Jacobien du résidu par rapport aux paramètres
    def R_flat(params):
        return residual_fn(params, model, t)

    # jacrev -> dérivée wrt params
    J_dict = jacrev(R_flat)(params)

    # flatten chaque gradient
    J = []
    for key in J_dict:
        J.append(J_dict[key].reshape(t.shape[0], -1))
    J = torch.cat(J, dim=1)  # (N, P)

    # NTK = J J^T
    K = J @ J.T

    return K


# =========================
# 6. MAIN
# =========================
if __name__ == "__main__":
    torch.manual_seed(0)

    N = 50
    t = torch.linspace(0, 1, N).unsqueeze(1)

    model = MLP()

    K = compute_ntk_fast(model, t)

    eigenvalues, eigenvectors = torch.linalg.eigh(K)

    print("Eigenvalues:")
    print(eigenvalues)

    # visualisation
    import matplotlib.pyplot as plt
    plt.semilogy(eigenvalues.detach().cpu())
    plt.title("NTK Spectrum (Optimized PINN)")
    plt.show()