import numpy as np
import tqdm

class GoyRK4:

    def __init__(
        self,
        N=22,
        k0=0.125,
        lmb=2.0,
        eps=0.5,
        nu=1e-7,
        dt=1e-5,
        force=0.005,
        N_force=4,
    ):

        self.N = N
        self.k0 = k0
        self.lmb = lmb
        self.eps = eps
        self.nu = nu
        self.dt = dt
        self.force = force
        self.N_force = N_force

        self.sh = k0 * lmb ** np.arange(N)

        self.A2 = np.full(N, -eps / lmb)
        self.A3 = np.full(N, -(1.0 - eps) / (lmb**2))

    # ============================================================
    # RHS GOY
    # ============================================================

    def rhs(self, X, Y):

        N = self.N
        sh = self.sh
        A2 = self.A2
        A3 = self.A3

        NX = np.zeros(N, dtype=np.float64)
        NY = np.zeros(N, dtype=np.float64)

        # --------------------------------------------------------
        # shell 0
        # --------------------------------------------------------

        NX[0] = X[2]*Y[1] + Y[2]*X[1]

        NY[0] = X[2]*X[1] - Y[1]*Y[2]

        # --------------------------------------------------------
        # shell 1
        # --------------------------------------------------------

        NX[1] = (
            X[3]*Y[2] + Y[3]*X[2]
            + A2[1]*(X[2]*Y[0] + Y[2]*X[0])
        )

        NY[1] = (
            X[3]*X[2] - Y[2]*Y[3]
            + A2[1]*(X[2]*X[0] - Y[2]*Y[0])
        )

        # --------------------------------------------------------
        # shells internes
        # --------------------------------------------------------

        for k in range(2, N-2):

            NX[k] = (
                X[k+2]*Y[k+1] + Y[k+2]*X[k+1]
                + A2[k]*(X[k+1]*Y[k-1] + Y[k+1]*X[k-1])
                + A3[k]*(X[k-1]*Y[k-2] + Y[k-1]*X[k-2])
            )

            NY[k] = (
                X[k+2]*X[k+1] - Y[k+1]*Y[k+2]
                + A2[k]*(X[k+1]*X[k-1] - Y[k+1]*Y[k-1])
                + A3[k]*(X[k-1]*X[k-2] - Y[k-1]*Y[k-2])
            )

        # --------------------------------------------------------
        # shell N-2
        # --------------------------------------------------------

        NX[N-2] = (
            A2[N-2]*(X[N-1]*Y[N-3] + Y[N-1]*X[N-3])
            + A3[N-2]*(X[N-3]*Y[N-4] + Y[N-3]*X[N-4])
        )

        NY[N-2] = (
            A2[N-2]*(X[N-1]*X[N-3] - Y[N-1]*Y[N-3])
            + A3[N-2]*(X[N-3]*X[N-4] - Y[N-3]*Y[N-4])
        )

        # --------------------------------------------------------
        # shell N-1
        # --------------------------------------------------------

        NX[N-1] = (
            A3[N-1]*(X[N-2]*Y[N-3] + Y[N-2]*X[N-3])
        )

        NY[N-1] = (
            A3[N-1]*(X[N-2]*X[N-3] + Y[N-2]*Y[N-3])
        )

        # facteur k_n
        NX *= sh
        NY *= sh

        # viscosité
        visc = self.nu * sh**2

        dXdt = NX - visc * X
        dYdt = NY - visc * Y

        # forcing
        dXdt[self.N_force] += self.force

        return dXdt, dYdt

    # ============================================================
    # RK4 step
    # ============================================================

    def rk4_step(self, X, Y):

        dt = self.dt

        k1x, k1y = self.rhs(X, Y)

        k2x, k2y = self.rhs(
            X + 0.5*dt*k1x,
            Y + 0.5*dt*k1y
        )

        k3x, k3y = self.rhs(
            X + 0.5*dt*k2x,
            Y + 0.5*dt*k2y
        )

        k4x, k4y = self.rhs(
            X + dt*k3x,
            Y + dt*k3y
        )

        Xnew = X + (dt/6.0)*(k1x + 2*k2x + 2*k3x + k4x)

        Ynew = Y + (dt/6.0)*(k1y + 2*k2y + 2*k3y + k4y)

        return Xnew, Ynew

    # ============================================================
    # intégration
    # ============================================================

    def integrate(self, X0, Y0, n_fs):

        X = X0.copy()
        Y = Y0.copy()

        # Xhist = np.zeros((n_steps, self.N))
        # Yhist = np.zeros((n_steps, self.N))

        for i in range(n_fs):

            X, Y = self.rk4_step(X, Y)

            # Xhist[i] = X
            # Yhist[i] = Y

        return X, Y

    # ============================================================
    # init
    # ============================================================

    def init_fields(self):

        X0 = self.sh ** (-1.0 / 3.0)

        Y0 = np.full(self.N, 1e-4)

        return X0, Y0


# ================================================================
# Exemple
# ================================================================

if __name__ == "__main__":

    model = GoyRK4()
    N_FS = int(1/model.dt/100)
    X0, Y0 = model.init_fields()
    Xhist = np.zeros((100, model.N))
    Yhist = np.zeros((100, model.N))
    for i in tqdm.tqdm(range(100)):
        X, Y = model.integrate(
            X0,
            Y0,
            n_fs=N_FS
        )
        Xhist[i] = X
        Yhist[i] = Y
  
    data_real_imag = np.empty((Xhist.shape[0], 2*Xhist.shape[1]), dtype=np.float64)
    PATH = "/home/s26calme/Documents/code_stage/KF/"
    np.save(PATH + "goy_rk4.npy", data_real_imag)
    # énergie totale finale
   # energy = 0.5 * np.sum(np.abs(data[-1]) ** 2)

    print("Simulation terminée")
    #print("Énergie finale :", energy)