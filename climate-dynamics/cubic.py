import numpy as np
import matplotlib.pyplot as plt

def cubic_splines(x, xnodes, data):
    n = len(xnodes) - 1
    h = np.diff(xnodes)  # Spacing between each pair of nodes

    # Set up the coefficient matrix for the system to solve for sigmas
    # The matrix is tridiagonal with variable coefficients depending on h
    A = np.zeros((n-1, n-1))
    np.fill_diagonal(A, 2 * (h[:-1] + h[1:]))  # Main diagonal
    np.fill_diagonal(A[1:], h[1:-1])           # Upper diagonal
    np.fill_diagonal(A[:, 1:], h[1:-1])        # Lower diagonal

    # g contains the RHS of the linear system
    g = 6 * ((data[2:] - data[1:-1]) / h[1:] - (data[1:-1] - data[:-2]) / h[:-1])

    # Solve the system to find sigmas
    sigmas = np.linalg.solve(A, g)
    sigmas = np.concatenate(([0], sigmas, [0]))  # Boundary conditions for natural spline

    # Find alphas and betas
    betas = (data[:-1] - sigmas[:-1] * h**2 / 6) / h
    alphas = (data[1:] - sigmas[1:] * h**2 / 6) / h

    # Find the indices of the two nodes closest to x
    indices = np.argsort(abs(xnodes - x))
    vals = indices[:2]
    ordered_vals = np.sort(xnodes[vals])
    idx = np.where(xnodes == ordered_vals[0])[0]
    idx1 = np.where(xnodes == ordered_vals[1])[0]

    # Calculate the cubic spline polynomial coefficients for the interval
    x1 = xnodes[idx1] - x
    x_1 = x - xnodes[idx]

    # Evaluate the cubic spline using the coefficients
    y_cubic = ((x1**3 * sigmas[idx] + x_1**3 * sigmas[idx1]) / (6 * h[idx]) +
               alphas[idx] * x_1 +
               betas[idx] * x1)

    return y_cubic


