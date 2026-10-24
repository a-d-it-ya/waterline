"""Truth trajectory + noisy altimeter for a descending aircraft over a rough sea."""
import numpy as np
import matplotlib.pyplot as plt
from waves import sea_surface


def simulate_descent(hs, rng, dt=0.02, t_end=18.0, h0=40.0, v_nom=-2.0,
                     turb_sigma=0.3, turb_tau=3.0, alt_hz=25, alt_sigma=0.1):
    t = np.arange(0, t_end, dt)
    n = len(t)
    eta = sea_surface(hs, t, rng)                    # water surface height [m]

    # vertical speed = nominal descent + gusty turbulence (first-order Gauss-Markov)
    phi = np.exp(-dt / turb_tau)
    turb = np.zeros(n)
    for k in range(n - 1):
        turb[k + 1] = phi * turb[k] + np.sqrt(1 - phi**2) * turb_sigma * rng.standard_normal()
    v = v_nom + turb
    h = h0 + np.concatenate(([0.0], np.cumsum(v[:-1] * dt)))   # height above MEAN water level

    # altimeter: height above the actual moving surface, noisy, sampled at alt_hz
    step = round(1 / (alt_hz * dt))
    alt = np.full(n, np.nan)                         # NaN = no measurement at this step
    idx = np.arange(0, n, step)
    alt[idx] = h[idx] - eta[idx] + rng.normal(0, alt_sigma, len(idx))

    return dict(t=t, eta=eta, h=h, v=v, alt=alt, dt=dt, alt_sigma=alt_sigma)


if __name__ == "__main__":
    run = simulate_descent(1.0, np.random.default_rng(1))
    plt.plot(run["t"], run["h"], label="true height above mean level")
    plt.plot(run["t"], run["h"] - run["eta"], label="true height above surface", alpha=0.7)
    plt.plot(run["t"], run["alt"], ".", ms=3, label="altimeter")
    plt.xlabel("time [s]"); plt.ylabel("height [m]"); plt.legend(); plt.grid(True)
    plt.show()
    plt.plot(run["t"], run["alt"] - run["h"], ".", ms=3)
    plt.xlabel("time [s]"); plt.ylabel("altimeter minus true height above mean level [m]")
    plt.grid(True); plt.show()