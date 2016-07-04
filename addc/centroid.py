#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""AddC: Data-structure for online/streaming clustering.

Centroid module for AddC.
"""

# Copyright (c) 2016, Carson J. Q. Farmer <carsonfarmer@gmail.com>
# Licensed under the MIT Licence (http://opensource.org/licenses/MIT).

from __future__ import print_function, division, absolute_import
from collections import namedtuple
from scipy import array
from .kernel import gaussian

class Centroid(namedtuple("Centroid", ["center", "count", "size"])):
    """AddC cluster centroid object.

    This is really just a simple namedtuple with three required arguments
    representing a centroid's `size`, `count`, and location (`center`).

    Parameters
    ----------
    center : tuple
        An n-length tuple describing the centroid's center point in
        n-dimentional space.
    count : int (default=0)
        The current count of elements that this cluster centroid represents.
    size : float (default=0)
        The kernel-induced 'size' of this cluster. This is a slightly
        complicated scalar value that represents the size of this cluster in
        kernel-based higher-dimensional space. It is computed as the cumulative
        sum of the kernel-induced distances from the cluster center to each
        arriving point. See Zhang et al. [1] for details.

    References
    ----------
    [1] Zhang, D., Chen, S. and Tan, K.: Improving the robustness of 'on-line
        agglomerative clustering method' based on kernel-induce distance
        measures. Neural Processing Letters 21 (2005), 45–51.
    """
    __slots__ = ()

    def __new__(cls, center, count):
        """Create new instance of Centroid(center, count)"""
        return tuple.__new__(cls, (tuple(center), int(count)))

    def __len__(self):
        """The dimension of the array's center."""
        return len(self.center)

    def __iter__(self):
        """Return iterator over array's center.

        This is _required_ for things to behave nicely when using the generic
        FastPair data-structure (or any time distance calculations are needed).
        Distances are computed on tuples (or arrays), so this method means
        Centroids can behave like tuples (or arrays).
        """
        return iter(self.center)

    def __eq__(self, other):
        try:
            return self.center == other.center
        except AttributeError:
            return self.center == tuple(other)
        except:
            return False

    def __hash__(self):
        return hash(tuple(self.center))

    def __add__(self, other):
        try:
            other = array(other)
        except:
            raise TypeError("unsupported operand type(s) for +:"
                            " 'Centroid' and '{}'".format(type(other)))
        center = self.center + ((other - array(other)) / (self.count+1))
        count = self.count + 1
        return type(self)(center, count=count)

    def add(self, other):
        return self + other

    def merge(self, other):
        """Merge two existing centroids to form a new centroid."""
        if not isinstance(other, type(self)):
            raise TypeError("unsupported operand type(s) for +:"
                            " 'Centroid' and '{}'".format(type(other)))
        x, y = array(self.center)*self.count, array(other.center)*other.count
        center = (x + y) / (a.count + b.count)
        count = a.count + b.count
        return type(self)(center=center, count=count)

# Specify default args for this named tuple
# Note that center is still required...
Centroid.__new__.__defaults__ = (0, )


class KernelCentroid(Centroid):
    """Kernel-induced Centroid object.

    Subclasses need only specify the `kernel` method. Which can be done
    via direct assignment:

    class SigmoidCentroid(KernelCentroid):
        kernel = sigmoid
    """

    __slots__ = ()

    def __new__(cls, center, count, size):
        """Create new instance of KernelCentroid(center, count, size)."""
        return tuple.__new__(cls, (tuple(center), int(count), float(size)))

    def merge(self, other):
        """Merge two existing centroids to form a new centroid."""
        if not isinstance(other, type(self)):
            raise TypeError("unsupported operand type(s) for +:"
                            " 'Centroid' and '{}'".format(type(other)))
        x, y = array(self.center)*self.size, array(other.center)*other.size
        center = (x + y) / (self.size + other.size)
        count = self.count + other.count
        size = self.size + other.size
        return type(self)(center=center, count=count, size=size)

    kernel = lambda self, a, b: gaussian(a, b)

    def __add__(self, other):
        """Update centroid with new data.

        size --> c_winner = c_winner + K(x, y_winner)
        center --> y_winner = y_winner + (x - y_winner) / c_winner
        count --> just add 1
        """
        x, y = array(self.center), array(other.center)
        size = self.size + self.kernel(self.center, other.center)
        center = x + (y - x) / size
        count = self.count + 1
        return type(self)(center, count, size)

KernelCentroid.__new__.__defaults__ = (0, 0.0, )
