"""Simplified sea-surface model: random sum of sinusoids.

Assumptions:
- Long-crested waves, treated as a time series at the aircraft's position.
- Periods 3-10 s, random phases, amplitudes scaled so std(eta) ~ Hs/4.
- No wave-aircraft interaction and no Doppler shift from forward speed (yet).
"""
import numpy as np
import matplotlib.pyplot as plt


def sea_surface(hs, t, rng, n_waves=8):
    """Water surface height eta(t) [m] above mean level for significant wave height hs [m]."""
    periods = rng.uniform(3, 10, n_waves)            # each wave gets its own period [s]
    omega = 2 * np.pi / periods                      # angular frequencies [rad/s]
    amps = rng.uniform(0.5, 1.0, n_waves)            # raw sizes, rescaled below
    amps *= (hs / 4) / np.sqrt(np.sum(amps**2) / 2)  # std(eta) = sqrt(sum(A^2)/2) = hs/4
    phases = rng.uniform(0, 2 * np.pi, n_waves)      # random starting points
    waves = amps[:, None] * np.sin(omega[:, None] * t[None, :] + phases[:, None])
    return waves.sum(axis=0)                         # superposition of all waves


if __name__ == "__main__":
    hs = 1.0
    t = np.arange(0, 600, 0.02)                      # 600 s at 50 Hz

    # Check 1: the roughness matches hs across several random seeds
    for seed in range(5):
        eta = sea_surface(hs, t, np.random.default_rng(seed))
        print(f"seed {seed}: std = {np.std(eta):.3f}   expected ~ {hs / 4:.3f}")

    # Check 2: plot the first 60 s
    eta = sea_surface(hs, t, np.random.default_rng(1))
    plt.plot(t[:3000], eta[:3000])
    plt.xlabel("time [s]")
    plt.ylabel("eta [m]")
    plt.title(f"Simulated sea surface, Hs = {hs} m")
    plt.grid(True)
    plt.show()