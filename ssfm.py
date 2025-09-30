import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
import warnings

warnings.filterwarnings("ignore")

# Parametri modela
alpha = 1.4
sigma = 0.7
beta = 0.08

# Diskretizacija prostora
N = 256  # rezolucija mreže
L = 50.0  # dužina domena u x i y ([-L/2, L/2])
dx = L / N
x = np.linspace(-L / 2, L / 2 - dx, N)
y = np.linspace(-L / 2, L / 2 - dx, N)
X, Y = np.meshgrid(x, y, indexing='xy')
R = np.sqrt(X ** 2 + Y ** 2)

# Talasni vektor koji odgovara Furije transformu Δ⊥
kx = 2 * np.pi * np.fft.fftfreq(N, d=dx)
ky = 2 * np.pi * np.fft.fftfreq(N, d=dx)
KX, KY = np.meshgrid(kx, ky, indexing='xy')
K2 = KX ** 2 + KY ** 2
K = np.sqrt(K2)

# R̃(k)
R_tilde = 2 * np.pi * alpha * sigma ** 2 / (1.0 + (sigma ** 2) * K2) - 2 * np.pi / (1.0 + K2) - beta
R_tilde_0 = 2 * np.pi * alpha * sigma ** 2 - 2 * np.pi - beta

# Potencijal talasovoda
U = (R / 4.0) ** 2


# SSFM korak
def ssfm_step(psi, dz):
    # 1) polu-korak potencijal
    psi = np.exp(-1j * U * (dz / 2.0)) * psi
    # 2) linearni korak

    psi_k = np.fft.fft2(psi)
    psi_k *= np.exp(-1j * K2 * dz)
    psi = np.fft.ifft2(psi_k)

    # 3) nelinearni korak
    I = np.abs(psi) ** 2
    Ik = np.fft.fft2(I)
    conv = np.fft.ifft2(R_tilde * Ik).real  # (R * |ψ|^2)(r)
    Nloc = -conv  # N(r)
    psi *= np.exp(-1j * Nloc * dz)
    # 4) polu-korak potencijal
    psi = np.exp(-1j * U * (dz / 2.0)) * psi
    return psi


# Incijalizacija Tomas-Fermi profila
rng = np.random.default_rng(1)
def init_TF(I):
    w = 4.0 * np.sqrt(-I * R_tilde_0)
    profile = np.sqrt(I) * np.sqrt(np.maximum(0.0, 1.0 - (R ** 2) / (w ** 2)))
    noise = 0.005 * np.sqrt(I) * (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N)))
    psi0 = profile + noise
    return psi0, w

# Propagacija duz z-ose
def propagate(I, zmax=10.0, dz=0.05):
    psi, w = init_TF(I)
    steps = int(np.round(zmax / dz))
    for step in range(steps):
        psi = ssfm_step(psi, dz)
    return psi


# I < I_MI; I > I_MI
I_lo = 10.0
I_hi = 20.0

psi_lo = propagate(I_lo, zmax=10.0, dz=0.005)
psi_hi = propagate(I_hi, zmax=10.0, dz=0.005)

amp_lo = np.abs(psi_lo)
amp_hi = np.abs(psi_hi)
phase_hi = np.angle(psi_hi)


# Plotovanje rezultata
def plot_intensity(field, title, nticks=5, phase_flag=False):
    plt.figure(figsize=(10, 9))

    label = "|$\psi|/max_r(\psi)$"
    if phase_flag:
        label = "arg[$\psi$]"
    ax = sns.heatmap(
        field,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={'label': label},
        cmap="jet"
    )

    ax.invert_yaxis()

    xticks = np.linspace(0, field.shape[1] - 1, nticks, dtype=int)
    yticks = np.linspace(0, field.shape[0] - 1, nticks, dtype=int)

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    ax.set_xticklabels(np.round(np.linspace(-L / 2, L / 2, nticks), 1))
    ax.set_yticklabels(np.round(np.linspace(-L / 2, L / 2, nticks), 1))

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)

    plt.tight_layout()
    plt.show()
    # plt.close()


plot_intensity(amp_lo ** 2 / (amp_lo ** 2).max(), f'Intenzitet posle z=10 (I={I_lo})', )
plot_intensity(amp_hi ** 2 / (amp_hi ** 2).max(), f'Intenzitet posle z=10 (I={I_hi})', )
plot_intensity(phase_hi, f'Faza posle z=10 (I={I_hi})', phase_flag=True)

def make_intensity_gif(I_min=10, I_max=20, nframes=10, zmax=10.0, dz=0.05, fname="intensity.gif"):
    """
    Generate a GIF of final intensity profiles as the initial intensity I
    increases from I_min to I_max.
    """
    frames = []
    intensities = np.linspace(I_min, I_max, nframes)

    for I in intensities:
        psi = propagate(I, zmax=zmax, dz=dz)
        amp = np.abs(psi)
        field = amp ** 2 / (amp ** 2).max()

        plt.figure(figsize=(10, 9))
        ax = sns.heatmap(
            field,
            xticklabels=False,
            yticklabels=False,
            cbar=False,
            cmap="jet",
            vmin=0, vmax=1
        )
        ax.invert_yaxis()
        ax.set_title(f"I = {I:.2f}, z = {zmax}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        plt.tight_layout()

        # Save current frame to memory buffer
        plt.savefig("frame.png", dpi=100)
        plt.close()
        frames.append(imageio.imread("frame.png"))

    # Write to GIF
    imageio.mimsave(fname, frames, duration=3)

    print(f"GIF saved as {fname}")

make_intensity_gif(I_min=10, I_max=20, nframes=25, fname="intensity.gif")

