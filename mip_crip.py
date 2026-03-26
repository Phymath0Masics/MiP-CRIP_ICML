"""
Ising Solvers - MiP-CRIP implementation
Minima Preserving Continuous Relaxation for Ising Problems
"""

import numpy as np


def MiP_CRIP(
    J, x_init, T=200, K=10,
    alpha=None, beta=None, lambda_=None,
    step=0.01,    # Adam learning rate
    beta1=0.9, beta2=0.999, eps=1e-8,
    sigma_noise=1e-3, u=None,
    rng=None, return_all=False
):
    """
    MiP-CRIP: Minima Preserving Continuous Relaxation for Ising Problems
    
    pCQO-MIS with Adam instead of momentum gradient descent.

    Minimizes over the box [-lambda_, lambda_]^n the energy
        H(x) = (beta/4) * ||x||_4^4 - (1/2) x^T (J + alpha I) x
    using Adam, then thresholds the final iterate to ±1 to get spin vector s.

    Parameters
    ----------
    J : (n, n) numpy array
        Symmetric coupling matrix with zero diagonal
    x_init : (n,) numpy array
        Initial point
    K : int
        Outer iterations (epochs)
    T : int
        Inner iterations (optimization steps per epoch)
    alpha : float
        Model parameter (must be provided)
    beta : float
        Model parameter (must be provided)
    lambda_ : float
        Model parameter (must be provided)
    step : float
        Adam learning rate
    beta1 : float
        Adam momentum parameter
    beta2 : float
        Adam momentum parameter
    eps : float
        Adam numerical stabilizer
    sigma_noise : float
        Variance for Gaussian sampling of next x_init
    u : optional
        One-flip local minima test flag
    rng : np.random.Generator or None
        Random number generator
    return_all : bool
        If True, also return S_opt list

    Returns
    -------
    spin_star : (n,) int array in {-1, +1}
        Optimal spin configuration
    S_opt : list (optional)
        List of candidate spin vectors (only if return_all=True)
    """

    J = np.asarray(J, dtype=float)
    n = J.shape[0]
    if x_init.shape[0] != n:
        raise ValueError("x_init length mismatch with J")

    if alpha is None or beta is None or lambda_ is None:
        raise ValueError("alpha, beta, lambda_ must be provided")

    if rng is None:
        rng = np.random.default_rng()

    A_alpha = J + alpha * np.eye(n)

    x_in = x_init.astype(float).copy()
    S_opt = []
    best_energy = -np.inf
    best_spin = None

    for outer in range(K):
        x = x_in.copy()

        # Adam state
        m = np.zeros_like(x)
        v = np.zeros_like(x)

        # ----- inner Adam loop -----
        for t in range(1, T + 1):
            # gradient: g(x) = beta * x^3 - (J + alpha I) x
            g_x = beta * (x**3) - A_alpha.dot(x)

            # Adam updates
            m = beta1 * m + (1.0 - beta1) * g_x
            v = beta2 * v + (1.0 - beta2) * (g_x * g_x)
            m_hat = m / (1.0 - beta1**t)
            v_hat = v / (1.0 - beta2**t)

            # gradient step + projection onto [-lambda_, lambda_]^n
            x = x - step * m_hat / (np.sqrt(v_hat) + eps)
            x = np.clip(x, -lambda_, lambda_)

        # ----- threshold to ±1 spins at radius λ -----
        s_T = np.where(x >= 0, 1, -1).astype(int)
        y_T = lambda_ * s_T

        if u is not None:
            g_y = beta * (y_T.astype(float)**3) - A_alpha.dot(y_T.astype(float))
        
            # (one-flip local minima test)
            if np.allclose(np.sign(g_y), -s_T, atol=1e-12):
                S_opt.append(s_T.copy())

        # energy of spin vector (original discrete Ising objective)
        energy = float(s_T.T.dot(J).dot(s_T))
        if energy > best_energy:
            best_energy = energy
            best_spin = s_T.copy()

        # sample next x_in ~ N(s_T, σ I)
        if sigma_noise < 0:
            raise ValueError("sigma_noise must be non-negative")
        if sigma_noise == 0:
            x_in = s_T.astype(float).copy()
        else:
            x_in = rng.normal(loc=y_T.astype(float),
                              scale=np.sqrt(sigma_noise),
                              size=n)

    # best element from S_opt vs global best
    if len(S_opt) == 0:
        spin_star = best_spin.copy()
    else:
        best_e = -np.inf
        best_s = None
        for s in S_opt:
            e = float(s.T.dot(J).dot(s))
            if e > best_e:
                best_e = e
                best_s = s.copy()
        spin_star = best_s if best_e >= best_energy else best_spin
    if return_all:
        return spin_star, S_opt
    return spin_star
