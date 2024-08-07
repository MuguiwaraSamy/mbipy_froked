"""
"""
from scipy.ndimage import gaussian_filter

def create_tikhonov_stack(xp):
    def tikhonov_stack(matrices, vectors, *, alpha):
        """
        Concatenate the input matrices and vectors with the regularisation terms:
        α¹ᐟ²Iₙ and 0ₙ respectively. As outlined in:
        https://math.stackexchange.com/a/299508.

        min ||(b) - (  A )x||
            ||(0)   (α¹ᐟ²I) ||_2

        Parameters
        ----------
        matrices : ArrayLike
            (..., M, N)
        vectors : ArrayLike
            (..., M)
        alpha : ArrayLike
            (...)

        Returns
        -------
        tuple[Array, Array]
            ((..., M + N, N), (..., M + N))
            Matrices concatenated with the regularisation matrix and vectors
            concatenated with zeros respectively.
        """
        matrices = xp.asarray(matrices)
        vectors = xp.asarray(vectors)
        alpha = xp.asarray(alpha)

        assert matrices.shape[:-1] == vectors.shape
        assert matrices.ndim == vectors.ndim + 1
        n = matrices.shape[-1]
        assert alpha.ndim <= matrices.ndim - 2

        alpha_05 = xp.sqrt(alpha).reshape(
            alpha.shape + (1,) * (matrices.ndim - alpha.ndim)
        )
        identity = xp.identity(n).reshape((1,) * (matrices.ndim - 2) + (n, n))

        tikhonov = xp.broadcast_to(
            identity * alpha_05,
            matrices.shape[:-2] + (n, n),
        )
        _matrices = xp.concatenate((matrices, tikhonov), axis=-2)
        _vectors = xp.concatenate(
            (vectors, xp.zeros(vectors.shape[:-1] + (n,))),
            axis=-1,
        )
        return _matrices, _vectors

    return tikhonov_stack


def create_normal_stack(xp):
    def normal_stack(matrices, vectors, *, alpha):
        """
        Form the stack of normal equations given stacks of matrices, vectors and
        regularisation factors.

        Parameters
        ----------
        matrices : ArrayLike
            (..., M, N)
        vectors : ArrayLike
            (..., M)
        alpha : ArrayLike
            (...)

        Returns
        -------
        tuple[Array, Array]
            ((..., N, N), (..., N))
            Stacks of the LHS and RHS of the normal equations respectively.
        """
        matrices = xp.asarray(matrices)
        vectors = xp.asarray(vectors)
        alpha = xp.asarray(alpha)

        alpha = alpha.reshape(alpha.shape + (1,) * (matrices.ndim - alpha.ndim))
        ata = xp.einsum("...ij, ...ik", matrices, matrices, optimize="optimal")
        atai = ata + xp.identity(matrices.shape[-1]) * alpha
        atb = xp.einsum("...ij, ...i", matrices, vectors, optimize="optimal")

        return atai, atb

    return normal_stack


def create_normal(xp):
    def normal(matrices, vectors):
        """
        Solve a stack of matrix equations (Ax = b).

        Parameters
        ----------
        matrices : ArrayLike
            (..., M, M)
            Stack of square matrices: A
        vectors : ArrayLike
            (..., M)
            Stack of vectors: b

        Returns
        -------
        Array
            (..., M)
            Stack of solutions: x
        """

        return xp.linalg.solve(matrices, vectors)

    return normal


def create_lstsq_solver(solvers, normal_stack, tikhonov_stack):
    def lstsq_solver(matrices, vectors, *, alpha=0.0, rcond=None, method=None):
        if method is None:
            method = "normal"
        if method not in solvers:
            raise ValueError(
                f"Invalid method: {method}. Valid methods: {solvers.keys()}"
            )

        if method in {"cholesky", "inv", "normal"}:
            matrices, vectors = normal_stack(matrices, vectors, alpha=alpha)

        elif method in {"lstsq", "pinv", "qr"}:
            matrices, vectors = tikhonov_stack(matrices, vectors, alpha=alpha)
            if method == "lstsq":
                return solvers[method](matrices, vectors, rcond=rcond)

        elif method in {"svd"}:
            return solvers[method](matrices, vectors, alpha=alpha, rcond=rcond)

        return solvers[method](matrices, vectors)

    return lstsq_solver


def create_implicit_tracking(xp, swv, solver):
    # TODO jax version
    def implicit_tracking(matrices, vectors, m, n, a=0, b=1, **kwargs):
        _m = 2 * m + 1
        _n = 2 * n + 1
        matrices = matrices[..., m:-m, n:-n, None, None, :, :]
        vectors = swv(vectors, (_m, _n), axis=(-3, -2)).transpose(
            tuple(range(vectors.ndim - 3)) + (-5, -4, -2, -1, -3)
        )
        vectors = vectors[..., a::b, a::b, :]
        result = solver(matrices, vectors, **kwargs)

        residuals = xp.einsum("...ij, ...j", matrices, result) - vectors

        minimum = (
            (residuals**2.0)
            .sum(axis=-1)
            .reshape(residuals.shape[:-3] + (-1,))
            .argmin(axis=-1)
        )
        minima = xp.unravel_index(minimum, residuals.shape[-3:-1])

        ndim = result.ndim - 4
        preceding = tuple(
            xp.arange(j).reshape((1,) * i + (-1,) + (1,) * (ndim - i))
            for i, j in enumerate(result.shape[:-3])
        )

        result_minimum = result[preceding + minima]

        # TODO nin17: check this
        result_minimum[..., 1] += b * minima[0] + (a - m)
        result_minimum[..., 2] += b * minima[1] + (a - n)

        return result_minimum

    return implicit_tracking


def create_laplace(xp):
    if xp.__name__ == "jax.numpy":

        def laplace(image_stack):
            output = xp.zeros_like(image_stack)
            output = output.at[..., 1:-1, :].add(
                image_stack[..., :-2, :]
                + image_stack[..., 2:, :]
                - (2 * image_stack[..., 1:-1, :])
            )
            output = output.at[..., 1:-1].add(
                image_stack[..., :-2]
                + image_stack[..., 2:]
                - (2 * image_stack[..., 1:-1])
            )
            output = output.at[..., 0, :].add(
                image_stack[..., 0, :]
                + image_stack[..., 1, :]
                - 2 * image_stack[..., 0, :]
            )
            output = output.at[..., -1, :].add(
                image_stack[..., -1, :]
                + image_stack[..., -2, :]
                - 2 * image_stack[..., -1, :]
            )
            output = output.at[..., 0].add(
                image_stack[..., 0] + image_stack[..., 1] - 2 * image_stack[..., 0]
            )
            output = output.at[..., -1].add(
                image_stack[..., -1] + image_stack[..., -2] - 2 * image_stack[..., -1]
            )
            return output

    else:

        def laplace(image_stack):
            output = xp.zeros_like(image_stack)
            output[..., 1:-1, :] += (
                image_stack[..., :-2, :]
                + image_stack[..., 2:, :]
                - (2 * image_stack[..., 1:-1, :])
            )
            output[..., 1:-1] += (
                image_stack[..., :-2]
                + image_stack[..., 2:]
                - 2 * image_stack[..., 1:-1]
            )
            output[..., 0, :] += (
                image_stack[..., 0, :]
                + image_stack[..., 1, :]
                - 2 * image_stack[..., 0, :]
            )
            output[..., -1, :] += (
                image_stack[..., -1, :]
                + image_stack[..., -2, :]
                - 2 * image_stack[..., -1, :]
            )
            output[..., 0] += (
                image_stack[..., 0] + image_stack[..., 1] - 2 * image_stack[..., 0]
            )
            output[..., -1] += (
                image_stack[..., -1] + image_stack[..., -2] - 2 * image_stack[..., -1]
            )
            return output

    return laplace


def prep_coloration(xp):
    def From_tensor_to_elipse(result_stack):
        """
        Transform tensor data into ellipse parameters and generate colored images.

        Parameters:
        - result_stack: Array containing the tensor data to be processed.
        - **kwargs: Additional keyword arguments for various parameters:
          - threshold: Threshold for clipping values. Default is 1.
          - sigma: Sigma value for the Gaussian filter in DDF_metrics. Default is 5.
          - epsilon: Small epsilon value to replace zeros and large outliers. Default is 1e-6.
          - nb_of_std: Number of standard deviations for normalization. Default is 3.
          - define_min: Boolean to define the minimum value based on mean - nb_of_std * std. Default is False.
          - input_range: Input range for rescaling intensity. Default is 'in_range'.
          - output_range: Output range for rescaling intensity. Default is 'out_range'.

        Returns:
        - result_stack: Processed tensor data with additional parameters.
        - colored_stack: Colored images generated from the tensor data.
        """
        
        
        threshold = kwargs.get('threshold', 1)
        sigma = kwargs.get('sigma', 5)
        epsilon = kwargs.get('epsilon', 1e-6)
        nb_of_std = kwargs.get('nb_of_std', 3)
        define_min = kwargs.get('define_min', False)
        input_range = kwargs.get('input_range', 'in_range')
        output_range = kwargs.get('output_range', 'out_range')
        
        a11 = result_stack[..., 3] * result_stack[..., 4]
        a22 = result_stack[..., 5] * result_stack[..., 4]
        a12 = 0.5 * (result_stack[..., 3] * result_stack[..., 5])

        theta = 0.5 * xp.arctan2(2 * a12, a11 - a22)

        a = xp.sqrt(xp.abs(a11 * xp.cos(theta)**2 + a22 * xp.sin(theta)**2 + 2 * a12 * xp.cos(theta) * xp.sin(theta)))
        b = xp.sqrt(xp.abs(a11 * xp.sin(theta)**2 + a22 * xp.cos(theta)**2 - 2 * a12 * xp.cos(theta) * xp.sin(theta)))

        mask = xp.logical_or(a11*a22 - a12**2 <= 0, a11*a22 <= 0)

        theta = xp.where(theta < 0, theta + xp.pi, theta)
        theta = xp.where(theta >= 2*xp.pi, theta - 2*xp.pi, theta)
        theta = xp.where(a < b, theta + xp.pi/2, theta)
        
        eccentricity = normalize_values(xp, xp.where(mask, 0, a-b), nb_of_std=nb_of_std, define_min=define_min)[..., None]
        
        area = normalize_values(xp, xp.pi * a * b, nb_of_std=nb_of_std, define_min=define_min)[..., None]
        
        
        result_stack = clip_values(xp, result_stack, threshold=threshold, epsilon=epsilon)
        intensity = normalize_values(xp, xp.sqrt(result_stack[..., 3]**2 + result_stack[..., 4]**2 + result_stack[..., 5]**2), 
                                     nb_of_std=nb_of_std, define_min=define_min)[..., None]
        
        
        result_stack = xp.concatenate((result_stack,
                                        theta[..., None],
                                        eccentricity,
                                        area,
                                        intensity), axis=-1)
        result_stack = DDF_metrics(xp, result_stack, sigma=sigma)
        
        
        colored_tensor = colored_image_generation(xp, result_stack[..., 3], result_stack[..., 5], result_stack[..., 4], input_range=input_range, output_range=output_range)
        colored_eccentricity = colored_image_generation(xp, result_stack[..., -1], result_stack[..., -2], result_stack[..., -5], input_range=input_range, output_range=output_range)
        colored_area = colored_image_generation(xp, result_stack[..., -1], result_stack[..., -2], result_stack[..., -4], input_range=input_range, output_range=output_range)
        colored_intensity = colored_image_generation(xp, result_stack[..., -1], result_stack[..., -2], result_stack[..., -3], input_range=input_range, output_range=output_range)

        colored_stack = xp.concatenate((colored_tensor[..., None], colored_eccentricity[..., None], colored_area[..., None], colored_intensity[..., None]), axis=-1)
       
        colored_stack = xp.concatenate((colored_tensor[..., None], colored_eccentricity[..., None], colored_area[..., None], colored_intensity[..., None]), axis=-1)

        
        return result_stack, colored_stack
        
    return From_tensor_to_elipse
                 
    
    
    
    
    

    
    
    
    
def normalize_values(xp,image, nb_of_std=3, define_min=False,):
    """
    Normalize the values of an image based on the mean and standard deviation.

    This function normalizes the values of the input image to the range [0, 1].
    The normalization is based on the mean and a specified number of standard deviations.
    Optionally, a minimum value can be defined based on the mean minus the standard deviations.

    Parameters:
    - xp: The array module (e.g., numpy, cupy) to use for array operations.
    - image: Array representing the image to be normalized.
    - nb_of_std: Number of standard deviations to use for the normalization range. Default is 3.
    - define_min: Boolean indicating whether to define the minimum value based on mean - nb_of_std * std. Default is False.

    Returns:
    - An array with normalized values in the range [0, 1].
    """
    mean = xp.mean(image)
    std = xp.std(image)
    min_value = mean - nb_of_std * std if define_min else 0
    max_value = mean + nb_of_std * std
    return xp.clip((image - min_value) / (max_value - min_value), 0, 1)


    
def clip_values(xp,results,threshold=1,epsilon=1e-6):
    """
    Clip the values of array in the results to handle outliers and zero values.

    Values are clipped based on a threshold and a small epsilon value to avoid zeros.

    Parameters:
    - xp: The array module (e.g., numpy, cupy) to use for array operations.
    - results: Array containing the tensor to be processed.
    - threshold: Threshold value for clipping. Default is 1.
    - epsilon: Small epsilon value to replace zeros and large outliers. Default is 1e-6.

    Returns:
    - An array with clipped values.
    """
    
    results[...,3] = xp.where(xp.logical_or(results[...,3] == 0, xp.abs(results[...,3]) > threshold), epsilon, results[...,3])
    results[...,5] = xp.where(xp.logical_or(results[...,5] == 0, xp.abs(results[...,5]) > threshold), epsilon, results[...,5])
    results[...,4] = xp.where(xp.logical_or(results[...,4] == 0, xp.abs(results[...,4]) > threshold), epsilon*xp.sign(results[...,4]), results[...,4])
    return results
    


def DDF_metrics(xp,results, sigma=5):
    """
    Compute the Directional Derivative Field metrics for an input tensor.

    This function calculates the saturation and corrected theta values from the input tensor.
    

    Parameters:
    - xp: The array module (e.g., numpy, cupy) to use for array operations.
    - results: Array containing the input theta, where the 7th component (index 6) is used for calculations.
    - sigma: Standard deviation for Gaussian filtering. Default is 5.

    Returns:
    - An array with the original tensor and the computed saturation and theta_corrected metrics appended.
    """
    padding_value = int(6 * xp.round(sigma))
    Df_theta_padded = xp.pad(results[...,6], padding_value, mode='reflect')
    
    wcos = gaussian_filter(xp.cos(2 * xp.where(Df_theta_padded == 0, xp.nan, Df_theta_padded)), sigma=sigma/xp.sqrt(2), mode='reflect')
    wsin = gaussian_filter(xp.sin(2 * xp.where(Df_theta_padded == 0, xp.nan, Df_theta_padded)), sigma=sigma/xp.sqrt(2), mode='reflect')
    
    wcos = wcos[padding_value:-padding_value, padding_value:-padding_value]
    wsin = wsin[padding_value:-padding_value, padding_value:-padding_value]
    
    saturation = xp.sqrt(wcos**2 + wsin**2)
    theta_corrected = 0.5*xp.arctan2(wsin, wcos)
    theta_corrected = xp.where(theta_corrected < 0, theta_corrected + xp.pi, theta_corrected)
    theta_corrected = xp.where(theta_corrected >= 2*xp.pi, theta_corrected - 2*xp.pi, theta_corrected)
    theta_corrected = xp.where(xp.logical_and(wcos == 0, wsin == 0), 0, theta_corrected)
    
    return xp.concatenate((results, saturation[...,None], theta_corrected[...,None]), axis=-1)

def rescale_intensity(xp,image, in_range='in_range', out_range='out_range'):
    """
    Rescale the intensity of an image to a specified range.

    This function rescales the intensity values of the input image to a specified output range.
    If no input or output range is provided, it defaults to the minimum and maximum of the image
    for input range and [0, 1] for output range.

    Parameters:
    - xp: The array module (e.g., numpy, cupy) to use for array operations.
    - image: Array representing the image whose intensity needs to be rescaled.
    - in_range: Tuple specifying the input range. Default is 'in_range', which means the range is determined from the input data.
    - out_range: Tuple specifying the output range. Default is 'out_range', which means the output values will be in the range [0, 1].

    Returns:
    - An array with intensity values rescaled to the specified output range.
    """
    
    if in_range == 'in_range':
        in_range = (xp.nanmin(image), xp.nanmax(image))
    if out_range == 'out_range':
        out_range = (0, 1.)
    image = xp.clip(image, in_range[0], in_range[1])
    return (image - in_range[0]) / (in_range[1] - in_range[0]) * (out_range[1] - out_range[0]) + out_range[0]


def hsv_to_rgb(xp,hsv):
    """
    From matplotlib.colors
    Convert hsv values to rgb.

    Parameters
    ----------
    hsv : (..., 3) array-like
       All values assumed to be in range [0, 1]

    Returns
    -------
    rgb : (..., 3) ndarray
       Colors converted to RGB values in range [0, 1]
    """
    hsv = xp.asarray(hsv)

    # check length of the last dimension, should be _some_ sort of rgb
    if hsv.shape[-1] != 3:
        raise ValueError("Last dimension of ixput array must be 3; "
                         "shape {shp} was found.".format(shp=hsv.shape))

    in_shape = hsv.shape
    hsv = xp.array(
        hsv, copy=False,
        dtype=xp.promote_types(hsv.dtype, xp.float32),  # Don't work on ints.
        ndmin=2,  # In case ixput was 1D.
    )

    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]

    r = xp.empty_like(h)
    g = xp.empty_like(h)
    b = xp.empty_like(h)

    i = (h * 6.0).astype(int)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    idx = i % 6 == 0
    r[idx] = v[idx]
    g[idx] = t[idx]
    b[idx] = p[idx]

    idx = i == 1
    r[idx] = q[idx]
    g[idx] = v[idx]
    b[idx] = p[idx]

    idx = i == 2
    r[idx] = p[idx]
    g[idx] = v[idx]
    b[idx] = t[idx]

    idx = i == 3
    r[idx] = p[idx]
    g[idx] = q[idx]
    b[idx] = v[idx]

    idx = i == 4
    r[idx] = t[idx]
    g[idx] = p[idx]
    b[idx] = v[idx]

    idx = i == 5
    r[idx] = v[idx]
    g[idx] = p[idx]
    b[idx] = q[idx]

    idx = s == 0
    r[idx] = v[idx]
    g[idx] = v[idx]
    b[idx] = v[idx]

    rgb = xp.stack([r, g, b], axis=-1)

    return rgb.reshape(in_shape)

def colored_image_generation(xp, hue, saturation, value, input_range='in_range', output_range='out_range'):
    """
    Generate an RGB image from hue, saturation, and value components.

    This function rescales the input hue, saturation, and value arrays to the specified input and output ranges,
    combines them into an HSV image, and then converts this HSV image to an RGB image.

    Parameters:
    - xp: The array module (e.g., numpy, cupy) to use for array operations.
    - hue: Array representing the hue component of the image.
    - saturation: Array representing the saturation component of the image.
    - value: Array representing the value (brightness) component of the image.
    - input_range: Range for input values. Default is 'in_range', which means the range is determined from the input data.
    - output_range: Range for output values. Default is 'out_range', which means the output values will be in the range [0, 1].

    Returns:
    - An array representing the RGB image.
    """
    
    hue_rescaled = rescale_intensity(xp, hue, in_range=input_range, out_range=output_range)
    saturation_rescaled = rescale_intensity(xp, saturation, in_range=input_range, out_range=output_range)
    value_rescaled = rescale_intensity(xp, value, in_range=input_range, out_range=output_range)
    
    hsv_image = xp.stack((hue_rescaled, saturation_rescaled, value_rescaled), axis=-1)
    return hsv_to_rgb(xp,hsv_image)