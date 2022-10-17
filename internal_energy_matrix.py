import numpy as np
def get_matrix(alpha, beta, gamma, num_points):
    """Return the matrix for the internal energy minimization.
    # Arguments
        alpha: The alpha parameter.
        beta: The beta parameter.
        gamma: The gamma parameter.
        num_points: The number of points in the curve.
    # Returns
        The matrix for the internal energy minimization. (i.e. A + gamma * I)
    """
    row = np.r_[
        (2 * alpha + 6 * beta),
        -(alpha + 4 * beta),
        beta,
        np.zeros(num_points - 5),
        beta,
        -(+alpha + 4 * beta)
    ]
    A = np.zeros((num_points, num_points))
    for i in range(num_points):
        A[i] = np.roll(row, i)
    I=np.eye(num_points)
    M=np.linalg.inv(A+(I*gamma))
    return M