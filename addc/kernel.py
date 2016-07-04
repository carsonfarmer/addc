#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""AddC: Data-structure for online/streaming clustering.

Kernel module for AddC.
"""

# Copyright (c) 2016, Carson J. Q. Farmer <carsonfarmer@gmail.com>
# Licensed under the MIT Licence (http://opensource.org/licenses/MIT).

from __future__ import division, absolute_import, print_function
from math import exp, tanh, sqrt, pi as PI, acos, e as E
import scipy.spatial.distance as dist
from scipy import dot
from functools import partial


def linear(x, y, c=0):
    """Compute a linear kernel.

    The Linear kernel is the simplest kernel function. It is given by the
    inner product <x,y> plus an optional constant `c`:
                        K(x, y) = x'y + c
    where `x` and `y` are vectors in the input space (i.e., vectors of
    features computed from training or test samples), and `c` ≥ 0 is a free
    parameter (default=0). Kernel algorithms using a linear kernel are often
    equivalent to their non-kernel counterparts.
    """
    return dot(x, y) + c


def poly(x, y, a=1, c=0, d=2):
    """Compute feature-space similarity over polynomials of the input vectors.

    For degree-`d` (default=2) polynomials, the kernel is defined as:
                        K(x, y) = (a(x'y) + c)^d
    where `x` and `y` are vectors in the input space (i.e., vectors of
    features computed from training or test samples), `c` ≥ 0 is a free
    parameter trading off the influence of higher-order versus lower-order
    terms in the polynomial, and `a` is an adjustable slope parameter. When
    `c`=0 (the default), the kernel is called homogeneous. The Polynomial
    kernel is a non-stationary kernel. Polynomial kernels are well suited for
    problems where all the training data is normalized.
    """
    return (dot(x, y)/a + c)**d


def gaussian(x, y, sigma=1):
    """Compute a Gaussian kernel.

    The Gaussian kernel is an example of radial basis function kernel. It can
    be define as:
                    K(x, y) = exp(-||x - y||^2 / 2σ^2)
    where `x` and `y` are vectors in the input space (i.e., vectors of
    features computed from training or test samples), ``||x - y||^2` is the
    squared Euclidean norm, and the adjustable parameter `sigma`
    plays a major role in the performance of the kernel, and should be
    carefully tuned to the problem at hand. If overestimated, the exponential
    will behave almost linearly and the higher-dimensional projection will
    start to lose its non-linear power. In the other hand, if underestimated,
    the function will lack regularization and the decision boundary will be
    highly sensitive to noise in training data.
    """
    return exp(-(dist.sqeuclidean(x, y)/2*sigma**2))


def exponential(x, y, sigma=1):
    """Compute an exponential kernel.

    The exponential kernel is closely related to the Gaussian kernel, with
    only the square of the norm left out. It is also a radial basis function
    kernel:
                    K(x, y) = exp(-||x - y|| / 2σ^2)
    where `x` and `y` are vectors in the input space (i.e., vectors of
    features computed from training or test samples), ``||x - y||` is the
    Euclidean norm, and the adjustable parameter `sigma` is used to adjust
    the kernel 'bandwidth'. It is important to note that the observations made
    about the `sigma` parameter for the Gaussian kernel also apply to the
    Exponential and Laplacian kernels.

    See Also
    --------
    gaussian
    """
    return exp(-(dist.euclidean(x, y) / 2*sigma**2))


def laplacian(x, y, sigma=1):
    """Compute a laplacian kernel.

    The Laplace kernel is completely equivalent to the exponential kernel,
    except for being less sensitive for changes in the `sigma` parameter.
    Being equivalent, it is also a radial basis function kernel:
                    K(x, y) = exp(-||x - y|| / σ)
    where `x` and `y` are vectors in the input space (i.e., vectors of
    features computed from training or test samples), ``||x - y||` is the
    Euclidean norm, and the adjustable parameter `sigma` is used to adjust
    the kernel 'bandwidth'. It is important to note that the observations made
    about the `sigma` parameter for the Gaussian kernel also apply to the
    Exponential and Laplacian kernels.
    """
    return exp(-(dist.euclidean(x, y) / sigma))


def sigmoid(x, y, alpha=None, c=-E):
    """Compute a hyperbolic tanget (or sigmoid) kernel.

    The Hyperbolic Tangent Kernel is also known as the Sigmoid Kernel and as
    the Multilayer Perceptron (MLP) kernel. The Sigmoid Kernel comes from the
    Neural Networks field, where the bipolar sigmoid function is often used
    as an activation function for artificial neurons:
                    K(x, y) = tanh(αx'y + c)
    where `x` and `y` are vectors in the input space (i.e., vectors of
    features computed from training or test samples), and tanh is the
    hyperbolic tangent function. There are two adjustable parameters in the
    sigmoid kernel, the slope `alpha` and the intercept constant `c`. A common
    value for alpha is 1/d, where d is the data dimension.

    References
    ----------
    [1] Lin, H. T. and Lin, C. J.: A study on sigmoid kernels for SVM and the
        training of non-PSD kernels by SMO-type methods. Web. Last accessed
        June 28 (2016). <http://www.csie.ntu.edu.tw/~cjlin/papers/tanh.pdf>
    """
    if alpha is None:
        alpha = 1 / len(x)
    return tanh(alpha*dot(x, y) + c)


def rational_quadratic(x, y, c=0):
    """Compute a rational quadratic kernel.

    The Rational Quadratic kernel is less computationally intensive than the
    Gaussian kernel and can be used as an alternative when using the Gaussian
    becomes too expensive:
                    K(x, y) = 1 - (||x - y||^2 / (||x - y||^2 + c))
    where `x` and `y` are vectors in the input space (i.e., vectors of
    features computed from training or test samples), ||x - y||^2 is the
    squared Euclidean norm, and `c` ≥ 0 is a free parameter (default=0).
    """
    d = dist.sqeuclidean(x, y)
    return 1 - d / (d + c)


def multiquadric(x, y, c=0):
    """Compute a multiquadric kernel.

    The Multiquadric kernel can be used in the same situations as the Rational
    Quadratic kernel. As is the case with the Sigmoid kernel, it is also an
    example of an non-positive definite kernel:
                    K(x, y) = sqrt(||x - y||^2 + c^2)
    where `x` and `y` are vectors in the input space (i.e., vectors of
    features computed from training or test samples), ||x - y||^2 is the
    squared Euclidean norm, and `c` ≥ 0 is a free parameter (default=0).
    """
    return sqrt(dist.sqeuclidean(x, y) + c**2)


def inverse_multiquadratic(x, y, c=0):
    """Compute an inverse multiquadric kernel.

    The Inverse Multi Quadric kernel. As with the Gaussian kernel, results
    in a kernel matrix with full rank, and thus forms a infinite dimension
    feature space.

    See Also
    --------
    multiquadric
    """
    return 1 / multiquadric(x, y, c)


def circular(x, y, sigma):
    """Compute a circular kernel.

    The circular kernel is used in geostatic applications. It is an example
    of an isotropic stationary kernel and is positive definite in ℜ^2:
    K(x, y) = 2/π arccos(-(||x - y|| / σ)) -
              2/π (||x - y|| / σ) sqrt(1 - (||x - y|| / σ)^2)
    if ||x - y|| < σ, zero otherwise.
    where `x` and `y` are vectors in the input space (i.e., vectors of
    features computed from training or test samples), ||x - y|| is the
    Euclidean norm, and sigma is a free parameter with no reasonable default.
    In other words, `sigma` should be defined *a priori* based on some
    geostatistical analysis, such as semi-variogram analysis.
    """
    pi2 = 2/PI
    norm_sigma = dist.euclidean(x, y) / sigma
    return pi2*acos(-norm_sigma) - pi2*norm_sigma*sqrt(1 - norm_sigma**2)


def kernel_dist(kernel=linear, **kw):
    """Generic kernel-induced distance metric.

    To use, `partially` apply this function with a kernel argument.

    Examples
    --------
    >>> from functools import partial
    >>> dist = partial(kernel_dist, kernel=sigmoid)
    """
    if kernel is None:  # Don't use a kernel!
        return dist.euclidean
    elif getattr(kernel, "__name__") == "gaussian":
        # We have a 'shortcut' for Gaussian kernels... this is kinda hacky
        # But maybe worth it given the speedup our shortcut gets us?
        return lambda x, y: 2 - 2*gaussian(x, y, kw.get("sigma", 1))
    else:
        kern = partial(kernel, **kw)
        return lambda x, y: kern(x, x) - 2*kern(x, y) + kern(y, y)
