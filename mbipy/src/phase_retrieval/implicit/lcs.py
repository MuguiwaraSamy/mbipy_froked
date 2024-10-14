"""_summary_
"""

__all__ = (
    "create_lcs_matrices",
    "create_lcs_vectors",
    "create_lcs",
    "create_lcs_df",
    "create_lcs_df_matrices",
    'create_lcs_ddf',
    'create_lcs_ddf_matrices',
    'create_lcs_ddf_vectors',
    'create_lcs_ddf_colored'
)

import importlib


def create_lcs_matrices(xp):
    def lcs_matrices(reference):
        assert reference.ndim >= 3
        gradient = xp.gradient(reference, axis=(-2, -1))
        matrices = xp.stack((reference, -gradient[0], -gradient[1]), axis=-1)
        order = tuple(range(0, matrices.ndim - 4)) + (-3, -2, -4, -1)
        return matrices.transpose(order)
        # return xp.moveaxis(matrices, -4, -2) # TODO nin17: see if transpose is faster

    return lcs_matrices


def create_lcs_vectors():
    def lcs_vectors(sample):
        assert sample.ndim >= 3
        return sample.transpose(tuple(range(0, sample.ndim - 3)) + (-2, -1, -3))

    return lcs_vectors


def create_lcs(
    lcs_matrices, lcs_vectors, solver, implicit_tracking, jax=False, numba=False
):
    if sum((jax, numba)) > 1:
        raise ValueError("Only one of jax or numba can be True")
    if not jax and not numba:

        def lcs(reference, sample, weak_absorption=True, m=None, n=None, **kwargs):
            matrices = lcs_matrices(reference)
            vectors = lcs_vectors(sample)

            kwargs = {"a": 1, "b": 2} | kwargs

            if all(i is not None for i in (m, n)):
                result = implicit_tracking(matrices, vectors, m, n, **kwargs)
            else:
                kwargs.pop("a", None)
                kwargs.pop("b", None)
                result = solver(matrices, vectors, **kwargs)

            if weak_absorption:
                return result
            result[..., 1:] /= result[..., :1]
            return result

        return lcs
    if jax:

        def lcs_jax(reference, sample, weak_absorption=True, **kwargs):
            matrices = lcs_matrices(reference)
            vectors = lcs_vectors(sample)
            result = solver(matrices, vectors, **kwargs)
            if weak_absorption:
                return result
            result = result.at[..., 1:].divide(result[..., :1])
            return result

    if numba:
        nb = importlib.import_module("numba")
        np = importlib.import_module("numpy")

        # TODO refactor to use guvectorize when it is supported in a jit function

        @nb.extending.register_jitable
        def lcs_numba(reference, sample, weak_absorption=True, alpha=0.0, **kwargs):
            assert reference.shape == sample.shape
            assert reference.ndim == 3
            x, y, z = reference.shape
            matrices = np.empty((y, z, x, 3), dtype=np.float64)
            out = np.empty((y, z, 3), dtype=np.float64)
            # TODO check this alpha stuff
            # alpha = np.asarray(alpha, dtype=np.float64)
            # alpha = alpha.reshape(alpha.shape + [1 for _ in range(matrices.ndim - alpha.ndim)])
            alpha_identity = alpha * np.identity(3, dtype=np.float64)

            sample = sample.transpose(1, 2, 0).copy()
            # TODO nin17: edges
            for j in nb.prange(1, y - 1):
                for k in range(1, z - 1):
                    for i in range(x):
                        matrices[j, k, i, 0] = reference[i, j, k]
                        matrices[j, k, i, 1] = (
                            -reference[i, j + 1, k] + reference[i, j - 1, k]
                        ) / 2.0
                        matrices[j, k, i, 2] = (
                            -reference[i, j, k + 1] + reference[i, j, k - 1]
                        ) / 2.0

            for j in nb.prange(1, y - 1):
                for k in range(1, z - 1):
                    a = matrices[j, k]
                    ata = (a.T @ a) + alpha_identity
                    atb = a.T @ sample[j, k]
                    out[j, k] = np.linalg.solve(ata, atb)

            if weak_absorption:
                return out

            for i in nb.prange(1, y - 1):
                for j in range(1, z - 1):
                    out[i, j, 1:] /= out[i, j, 0]

            return out

        return lcs_numba

    return lcs_jax


def create_lcs_df_matrices(xp, laplace):
    def lcs_matrices(reference):
        assert reference.ndim >= 3
        gradient = xp.gradient(reference, axis=(-2, -1))
        laplacian = laplace(reference)
        matrices = xp.stack((reference, -gradient[0], -gradient[1], laplacian), axis=-1)
        order = tuple(range(0, matrices.ndim - 4)) + (-3, -2, -4, -1)
        return matrices.transpose(order)
        # return xp.moveaxis(matrices, -4, -2) # TODO nin17: see if transpose is faster

    return lcs_matrices


def create_lcs_df(lcs_df_matrices, lcs_df_vectors, solver):
    def lcs(reference, sample, weak_absorption=True, m=None, n=None, **kwargs):

        matrices = lcs_df_matrices(reference)
        vectors = lcs_df_vectors(sample)

        kwargs = {"a": 1, "b": 2} | kwargs

        if all(i is not None for i in (m, n)):
            # result = implicit_tracking(matrices, vectors, m, n, **kwargs)
            pass
            # TODO nin17: lcs_df tracking
        else:
            kwargs.pop("a", None)
            kwargs.pop("b", None)
            result = solver(matrices, vectors, **kwargs)
        # result = solver(matrices, vectors, **kwargs)

        if weak_absorption:
            return result
        result[..., 1:] /= result[..., :1]
        return result

    return lcs


def create_lcs_ddf(lcs_ddf_matrices, lcs_ddf_vectors, solver):
    def lcs_ddf(reference, sample, weak_absorption=True, m=None, n=None, **kwargs):
        matrices = lcs_ddf_matrices(reference)
        vectors = lcs_ddf_vectors(sample)

        kwargs = {"a": 1, "b": 2} | kwargs

        if all(i is not None for i in (m, n)):
            # result = implicit_tracking(matrices, vectors, m, n, **kwargs)
            pass
            # TODO nin17: lcs_df tracking
        else:
            kwargs.pop("a", None)
            kwargs.pop("b", None)
            result = solver(matrices, vectors, **kwargs)
        # result = solver(matrices, vectors, **kwargs)

        if weak_absorption:
            return result
        result[..., 1:] /= result[..., :1]
        return result

    return lcs_ddf

def create_lcs_ddf_matrices(xp):
    def lcs_ddf_matrices(reference):
        assert reference.ndim >= 3
        gradient_x, gradient_y = xp.gradient(reference, axis=(-2, -1))
        gradient_xx, gradient_xy = xp.gradient(gradient_x,axis=(-2, -1))
        gradient_yx, gradient_yy = xp.gradient(gradient_y,axis=(-2, -1))
        matrices = xp.stack((reference, -gradient_x, -gradient_y, gradient_xx, gradient_yx, gradient_yy), axis=-1)
        order = tuple(range(0, matrices.ndim - 4)) + (-3, -2, -4, -1)
        return matrices.transpose(order)
        

    return lcs_ddf_matrices

def create_lcs_ddf_vectors():
    def lcs_ddf_vectors(sample):
        assert sample.ndim >= 3
        return sample.transpose(tuple(range(0, sample.ndim - 3)) + (-2, -1, -3))

    return lcs_ddf_vectors

def create_lcs_ddf_colored(lcs_ddf_matrices, lcs_ddf_vectors, solver, coloration):
    """
    Create a function to compute and colorize LCS DDF results.

    Parameters:
    - lcs_ddf_matrices: Function to compute matrices from the reference image.
    - lcs_ddf_vectors: Function to compute vectors from the sample image.
    - solver: Function to solve the matrices and vectors.
    - coloration: Function to apply coloration to the result.

    Returns:
    - lcs_ddf_colored: Function to compute and colorize LCS DDF for given reference and sample images.
    """
    def lcs_ddf_colored(reference, sample, weak_absorption=True, m=None, n=None, **kwargs):
        """
        Compute and colorize LCS DDF results for given reference and sample images.

        Parameters:
        - reference: The reference image.
        - sample: The sample image.
        - weak_absorption: Boolean to indicate if weak absorption should be considered. Default is True.
        - m: Optional parameter (not used in the current implementation).
        - n: Optional parameter (not used in the current implementation).
        - **kwargs: Additional parameters for the solver and coloration functions.
          - `threshold` (float): The threshold value used to filter out small or insignificant values in calculations.
          - `sigma` (float): The standard deviation for Gaussian filters used in smoothing operations.
          - `epsilon` (float): A small value used to prevent division by zero or handle very small values.
          - `nb_of_std` (int): The number of standard deviations used to normalize or scale values.
          - `define_min` (bool): Flag indicating whether to define the minimum value based on mean and standard deviation.
          - `input_range` (tuple or str): The range of input values for rescaling, or 'in_range' to use the minimum and maximum values from the input image.
          - `output_range` (tuple or str): The desired range of output values after rescaling, or 'out_range' to use the default range (0 to 1).

        Returns:
        - Colored LCS DDF results.
        """
        matrices = lcs_ddf_matrices(reference)
        vectors = lcs_ddf_vectors(sample)
        
        solver_kwargs = {}
        coloration_kwargs = {}

        for key, value in kwargs.items():
            if key.startswith('coloration_'):
                coloration_kwargs[key] = value
            else:
                solver_kwargs[key] = value

        result = solver(matrices, vectors, **solver_kwargs)
        # result = solver(matrices, vectors, **kwargs)
        
        #That what is done in the original code of the LCS dont know if it is necessary
        # result[..., 0] = 1 / result[..., 0]
        if weak_absorption:
            result
        else:
            result[..., 1:] /= result[..., :1]
        
        #That what is done in the original code of the LCS dont know if it is necessary
        # result[..., 0] = 1 / result[..., 0]
        return coloration(result, **coloration_kwargs)
        

    return lcs_ddf_colored
