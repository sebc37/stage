"""
GOY Shell Model - Python Implementation
Ported from C code originally by Nicolas B. Garnier (2022-07-05)

This module implements the GOY (Gledzer-Ohkitani-Yamada) shell model for studying
turbulence through a set of coupled ODEs.
"""

import numpy as np
import os
from pathlib import Path


# ============================================================================
#                            PARAMETERS
# ============================================================================

class GOYParams:
    """Configuration parameters for the GOY model."""
    
    # Model parameters
    force = 0.005              # strength of the random forcing
    N_force = 4                # mode which is forced
    force_rnd = True           # random forcing (True) or deterministic (False)
    k0 = 0.125                 # largest scale
    lmb = 2.0                  # ratio between consecutive scales
    eps = 0.5                  # for the NL coefficients
    nu = 1.0e-7                # viscosity
    N = 22                     # nb of modes
    
    # Scheme parameters
    dt = 1.0e-5                # for integration scheme
    fs = 100.                 # for data saving (frequency)
    time = 1.0e3               # total simulation time
    
    # Behavioral parameters
    DO_CONTINUE = False        # to continue the previous simulation
    DO_SAVE = True             # to save time trace of shells
    DO_FLUX = False            # to save time trace of fluxes between shells
    DO_MOMENTS = False         # to compute and save moments
    DO_STATS = False           # to compute additional statistics
    DO_RECONSTRUCT = False      # to output a "reconstructed" velocity signal
    DO_FIR = False             # to low-pass filter time traces (anti-aliasing)
    
    # File parameters
    MOMENTS_FILE = "moments_py2.dat"
    DATA_FILE = "data_py2.dat"
    RECONSTRUCTED_FILE = "v_py2.dat"
    FLUX_DATA_FILE = "flux_py2.dat"
    STATS_DIR = "stats"
    DATA_DIR = "."
    BACKUP_DIR = "tmp"


# ============================================================================
#                         GOY SHELL MODEL CLASS
# ============================================================================

class GOYShellModel:
    """The GOY shell model integrator."""
    
    def __init__(self, params=None):
        """Initialize the GOY shell model.
        
        Parameters
        ----------
        params : GOYParams, optional
            Model parameters. If None, uses default parameters.
        """
        self.p = params if params is not None else GOYParams()
        
        # Derived parameters
        self.N_fs = int(1.0 / (self.p.dt * self.p.fs))
        self.N_steps = 0
        
        # Initialize arrays
        self.sh = np.zeros(self.p.N)        # wave numbers
        self.X = np.zeros(self.p.N)         # real part of complex amplitude
        self.Y = np.zeros(self.p.N)         # imaginary part
        self.Xp = np.zeros(self.p.N)        # X at previous timestep
        self.Xpp = np.zeros(self.p.N)       # X at two timesteps ago
        self.Yp = np.zeros(self.p.N)        # Y at previous timestep
        self.Ypp = np.zeros(self.p.N)       # Y at two timesteps ago
        
        # Low-pass filtered versions (FIR)
        self.Xf = np.zeros(self.p.N)
        self.Yf = np.zeros(self.p.N)
        
        # Non-linear terms
        self.NXp = np.zeros(self.p.N)
        self.NXpp = np.zeros(self.p.N)
        self.NYp = np.zeros(self.p.N)
        self.NYpp = np.zeros(self.p.N)
        
        # Coefficients and forcing
        self.A = np.zeros(self.p.N)         # numerical scheme coefficients
        self.A1 = np.ones(self.p.N)         # NL coefficient (always 1)
        self.A2 = np.full(self.p.N, -self.p.eps / self.p.lmb)      # NL coefficient
        self.A3 = np.full(self.p.N, -(1.0 - self.p.eps) / (self.p.lmb**2))  # NL coefficient
        self.F = np.zeros(self.p.N)         # initial forcing profile
        self.D = np.zeros(self.p.N)         # forcing amplitude distribution
        
        # File handles for output
        self.data_file = None
        self.reconstructed_file = None
        self.flux_file = None
        
        # Initialize all parameters
        self._init_NL()
        self._init_forcing()
        self._init_shell()
    
    def _init_NL(self):
        """Initialize non-linear coefficients."""
        # Already done in __init__ via array creation
        pass
    
    def _init_forcing(self):
        """Initialize forcing distribution."""
        self.D[:] = 0.0
        self.D[self.p.N_force] = 1.0
    
    def _init_shell(self):
        """Initialize shell wave numbers and numerical scheme coefficients."""
        for i in range(self.p.N):
            self.sh[i] = self.p.k0 * (self.p.lmb ** i)
            self.A[i] = np.exp(-self.p.nu * self.sh[i]**2 * self.p.dt)
    
    def init_fields(self):
        """Initialize the fields with default initial conditions."""
        for i in range(self.p.N):
            self.F[i] = self.sh[i] ** (-1.0/3.0)
            self.Xpp[i] = self.F[i]
            self.Ypp[i] = 1.0e-4
        
        self.compute_NX(self.Xpp, self.Ypp, self.NXpp)
        self.compute_NY(self.Xpp, self.Ypp, self.NYpp)
        
        for i in range(self.p.N):
            self.Xp[i] = self.A[i] * (self.Xpp[i] + self.p.dt * self.NXpp[i])
            self.Yp[i] = self.A[i] * (self.Ypp[i] + self.p.dt * self.NYpp[i])
        
        self.compute_NX(self.Xp, self.Yp, self.NXp)
        self.compute_NY(self.Xp, self.Yp, self.NYp)
        
        if self.p.DO_FIR:
            self.reset_FIR(self.Xf)
            self.reset_FIR(self.Yf)
        
        self.N_steps = 0
    
    def reset_FIR(self, x):
        """Reset FIR filter array."""
        x[:] = 0.0
    
    def normalize_FIR(self, x):
        """Normalize FIR filter array."""
        x[:] /= self.N_fs
    
    def compute_NY_GOY(self, ax, ay):
        """Compute the Y component of the non-linear term for GOY model.
        
        Parameters
        ----------
        ax : ndarray
            X component (real part) at current timestep
        ay : ndarray
            Y component (imaginary part) at current timestep
        
        Returns
        -------
        res : ndarray
            Time derivative of Y
        """
        res = np.zeros(self.p.N)
        
        res[0] = ax[2] * ax[1] - ay[1] * ay[2]
        res[1] = (ax[3] * ax[2] - ay[2] * ay[3] + 
                  self.A2[1] * (ax[2] * ax[0] - ay[2] * ay[0]))
        
        for k in range(2, self.p.N - 2):
            res[k] = (ax[k+2] * ax[k+1] - ay[k+1] * ay[k+2] +
                      self.A2[k] * (ax[k+1] * ax[k-1] - ay[k+1] * ay[k-1]) +
                      self.A3[k] * (ax[k-1] * ax[k-2] - ay[k-1] * ay[k-2]))
        
        res[self.p.N - 2] = (self.A2[self.p.N - 2] * (ax[self.p.N - 1] * ax[self.p.N - 3] + 
                                                       ay[self.p.N - 1] * ay[self.p.N - 3]) +
                             self.A3[self.p.N - 2] * (ax[self.p.N - 3] * ax[self.p.N - 4] + 
                                                       ay[self.p.N - 3] * ay[self.p.N - 4]))
        
        res[self.p.N - 1] = self.A3[self.p.N - 1] * (ax[self.p.N - 2] * ax[self.p.N - 3] - 
                                                       ay[self.p.N - 2] * ay[self.p.N - 3])
        
        res *= self.sh
        return res
    
    def compute_NX_GOY(self, ax, ay):
        """Compute the X component of the non-linear term for GOY model.
        
        Parameters
        ----------
        ax : ndarray
            X component (real part) at current timestep
        ay : ndarray
            Y component (imaginary part) at current timestep
        
        Returns
        -------
        res : ndarray
            Time derivative of X
        """
        res = np.zeros(self.p.N)
        
        res[0] = ax[2] * ay[1] + ay[2] * ax[1]
        res[1] = (ax[3] * ay[2] + ay[3] * ax[2] + 
                  self.A2[1] * (ax[2] * ay[0] + ay[2] * ax[0]))
        
        for k in range(2, self.p.N - 2):
            res[k] = ((ax[k+2] * ay[k+1] + ay[k+2] * ax[k+1]) +
                      self.A2[k] * (ax[k+1] * ay[k-1] + ay[k+1] * ax[k-1]) +
                      self.A3[k] * (ax[k-1] * ay[k-2] + ay[k-1] * ax[k-2]))
        
        res[self.p.N - 2] = (self.A2[self.p.N - 2] * (ax[self.p.N - 1] * ay[self.p.N - 3] + 
                                                       ay[self.p.N - 1] * ax[self.p.N - 3]) +
                             self.A3[self.p.N - 2] * (ax[self.p.N - 3] * ay[self.p.N - 4] + 
                                                       ay[self.p.N - 3] * ax[self.p.N - 4]))
        
        res[self.p.N - 1] = self.A3[self.p.N - 1] * (ax[self.p.N - 2] * ay[self.p.N - 3] + 
                                                       ay[self.p.N - 2] * ax[self.p.N - 3])
        
        res *= self.sh
        return res
    
    def compute_NX(self, ax, ay, res):
        """Compute non-linear X term in-place."""
        res[:] = self.compute_NX_GOY(ax, ay)
    
    def compute_NY(self, ax, ay, res):
        """Compute non-linear Y term in-place."""
        res[:] = self.compute_NY_GOY(ax, ay)
    
    def integrate(self):
        """Perform one integration step using the numerical scheme."""
        if self.p.force_rnd:
            force1 = self.p.force * np.random.random()
            force2 = self.p.force * np.random.random()
        else:
            force1 = self.p.force
            force2 = self.p.force
        
        for i in range(self.p.N):
            self.X[i] = (self.A[i] * self.Xp[i] + 
                         1.5 * self.p.dt * self.A[i] * self.NXp[i] -
                         0.5 * self.p.dt * self.A[i]**2 * self.NXpp[i] +
                         self.p.dt * force1 * self.D[i])
            
            self.Y[i] = (self.A[i] * self.Yp[i] +
                         1.5 * self.p.dt * self.A[i] * self.NYp[i] -
                         0.5 * self.p.dt * self.A[i]**2 * self.NYpp[i] +
                         self.p.dt * force2 * self.D[i])
        
        if self.p.DO_FIR:
            self.Xf += self.X
            self.Yf += self.Y
    
    def setup_output_files(self):
        """Initialize output files."""
        Path(self.p.DATA_DIR).mkdir(exist_ok=True)
        
        # Reset data files
        open(os.path.join(self.p.DATA_DIR, self.p.DATA_FILE), 'w').close()
        open(os.path.join(self.p.DATA_DIR, self.p.RECONSTRUCTED_FILE), 'w').close()
        if self.p.DO_FLUX:
            open(os.path.join(self.p.DATA_DIR, self.p.FLUX_DATA_FILE), 'w').close()
    
    def save_data(self, my_X, my_Y):
        """Save shell amplitudes to data file."""
        filepath = os.path.join(self.p.DATA_DIR, self.p.DATA_FILE)
        with open(filepath, 'a') as f:
            for i in range(self.p.N):
                f.write(f"{my_X[i]:g} {my_Y[i]:g} ")
            f.write("\n")
    
    def save_reconstructed(self, my_X, my_Y, t):
        """Save reconstructed velocity field at wavenumber scale."""
        v = 0.0
        for i in range(self.p.N):
            v += my_X[i] * np.cos(self.sh[i] * t) - my_Y[i] * np.sin(self.sh[i] * t)
        
        filepath = os.path.join(self.p.DATA_DIR, self.p.RECONSTRUCTED_FILE)
        with open(filepath, 'a') as f:
            f.write(f"{v:g}\n")
    
    def run(self):
        """Run the full simulation."""
        count = int(self.p.time / self.p.dt)
        
        self.init_fields()
        self.setup_output_files()
        
        print(f"Starting GOY model integration: {count} steps ({self.p.time} time units)")
        print(f"Output interval: {self.N_fs} steps = {1.0/self.p.fs} time units")
        
        while count > 0:
            self.integrate()
            
            if (count % self.N_fs) == 0:
                # Print progress
                remaining_steps = int(count / self.N_fs)
                if not np.isnan(self.X[0]):
                    print(f"Remaining steps: {remaining_steps}")
                else:
                    print(f"Remaining steps: {remaining_steps} : ERROR (NaN detected)")
                
                # Save data
                if self.p.DO_FIR:
                    self.normalize_FIR(self.Xf)
                    self.normalize_FIR(self.Yf)
                    
                    if self.p.DO_SAVE:
                        self.save_data(self.Xf, self.Yf)
                    if self.p.DO_RECONSTRUCT:
                        self.save_reconstructed(self.Xf, self.Yf, self.N_steps / self.p.fs)
                    
                    self.reset_FIR(self.Xf)
                    self.reset_FIR(self.Yf)
                else:
                    if self.p.DO_SAVE:
                        self.save_data(self.X, self.Y)
                    if self.p.DO_RECONSTRUCT:
                        self.save_reconstructed(self.X, self.Y, self.N_steps / self.p.fs)
                
                self.N_steps += 1
            
            # Evolve fields
            self.Xpp[:] = self.Xp
            self.Ypp[:] = self.Yp
            self.NXpp[:] = self.NXp
            self.NYpp[:] = self.NYp
            self.Xp[:] = self.X
            self.Yp[:] = self.Y
            
            self.compute_NX(self.Xp, self.Yp, self.NXp)
            self.compute_NY(self.Xp, self.Yp, self.NYp)
            
            count -= 1
        
        print("Integration completed!")
        return np.loadtxt(os.path.join(self.p.DATA_DIR, self.p.DATA_FILE))


# ============================================================================
#                              MAIN PROGRAM
# ============================================================================

if __name__ == "__main__":
    # Create custom parameters if needed
    params = GOYParams()
    
    # Create and run the model
    model = GOYShellModel(params)
    data = model.run()
    
    print(f"\nData shape: {data.shape}")
    print(f"Data saved to: {os.path.join(params.DATA_DIR, params.DATA_FILE)}")
    
    # Display some statistics
    print(f"\nData statistics:")
    print(f"  Mean: {np.mean(data):.6e}")
    print(f"  Std Dev: {np.std(data):.6e}")
    print(f"  Min: {np.min(data):.6e}")
    print(f"  Max: {np.max(data):.6e}")
