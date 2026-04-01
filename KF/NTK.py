import torch
import torch.nn as nn
from torch.func import functional_call, vmap, jacrev

# --- GOY parameters ---
N_SHELLS = 22
LAMBDA   = 2.0
K0       = 1.0
DELTA    = 0.5
NU       = 1e-7

k = K0 * LAMBDA ** torch.arange(N_SHELLS).float()


def split_complex_interleaved(u_real):
    """(2N,) [Re1,Im1,Re2,Im2,...] -> (N,) complex"""
    return torch.complex(u_real[0::2], u_real[1::2])


def goy_rhs(u_c):
    """(N,) complex -> (N,) complex RHS of GOY model"""
    N = u_c.shape[0]

    def safe(n):
        if n < 0 or n >= N:
            return torch.zeros(1, dtype=u_c.dtype)
        return u_c[n].unsqueeze(0)

    rhs = []
    for n in range(N):
        term1 =  k[n]         * safe(n+1).conj() * safe(n+2).conj()
        term2 = -DELTA/2      * k[n-1] * safe(n-1).conj() * safe(n+1).conj() if n >= 1 else torch.zeros(1, dtype=u_c.dtype)
        term3 = -(1-DELTA)/4  * k[n-2] * safe(n-1).conj() * safe(n-2).conj() if n >= 2 else torch.zeros(1, dtype=u_c.dtype)

        rhs.append((1j * (term1 + term2 + term3) - NU * k[n]**2 * u_c[n]).squeeze())

    return torch.stack(rhs)


def residual_fn(params, model, t):
    """
    GOY residual in interleaved layout [Re(R1),Im(R1),Re(R2),Im(R2),...]
    Shape: (2N,)
    """
    def u_real_fn(t_scalar):
        return functional_call(model, params, (t_scalar.reshape(1, 1),)).squeeze()

    dudt = jacrev(u_real_fn)(t)                          # (2N,) interleaved

    u_c   = split_complex_interleaved(u_real_fn(t))      # (N,) complex
    rhs_c = goy_rhs(u_c)                                 # (N,) complex

    rhs = torch.zeros(2 * N_SHELLS)
    rhs[0::2] = rhs_c.real
    rhs[1::2] = rhs_c.imag

    return dudt - rhs                                    # (2N,)


def goy_residual_ntk(model, t_col):
    """
    Computes per-shell NTK eigenvalues and the full NTK matrix.

    Args:
        model : PINN with output layout [Re1,Im1,...,ReN,ImN]
        t_col : (M,) collocation times

    Returns:
        K_mat      : (M*2N, M*2N) full NTK matrix
        shell_eigs : dict {shell_index: eigenvalues tensor}
    """
    params = dict(model.named_parameters())

    def single_jac(t):
        jac_dict = jacrev(residual_fn, argnums=0)(params, model, t)
        return torch.cat([j.flatten(1) for j in jac_dict.values()], dim=1)  # (2N, P)

    J = vmap(single_jac)(t_col)                          # (M, 2N, P)

    # --- Full NTK matrix ---
    K_full = torch.einsum('iap,jbp->iajb', J, J)        # (M, 2N, M, 2N)
    K_mat  = K_full.reshape(len(t_col) * 2 * N_SHELLS, -1)

    # --- Per-shell NTK eigenvalues ---
    shell_eigs = {}
    for n in range(N_SHELLS):
        J_n    = J[:, [2*n, 2*n+1], :].reshape(len(t_col) * 2, -1)  # (2M, P)
        K_n    = J_n @ J_n.T
        shell_eigs[n] = torch.linalg.eigvalsh(K_n)

    return K_mat, shell_eigs


