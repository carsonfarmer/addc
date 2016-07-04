#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""AddC: Data-structure for online/streaming clustering of non-stationary data.

This module implements an on-line (streaming) agglomerative clustering
algorithm (AddC) for non-stationary data as described in Guedalia et al. [1],
with modifications as given in Zhang et al. [2]. Further improvements include
building the main AddC data-structure on top of the David Eppstein's dynamic
closest-pair algorithm/data-structure (FastPair) [3], and the ability to use
a variety of different kernel functions [4].

The AddC algorithm is simple and fast. The algorithm can be summarized in the
following three steps: For each data point arriving;
1. Move the closest centroid towards the point.
2. Merge the two closest centroids. This results in the creation of a
   redundant centroid.
3. Set the redundant centroid equal to the new data point.

The algorithm can be understood as follows: Three criteria are addressed at
each time step, minimization of the within cluster variance, maximization of
the distances between the centroids, and adaptation to temporal changes in
the distribution of the data. In the first step the within-cluster variance
is minimized by up-dating the representation in a manner similar to the well-
known k-means algorithm. The second step maximizes the distances between the
centroids by merging the two centroids with the minimum distance (not
considering their weight). The merging is similar to most agglomerative methods.
Finally non-stationarity in the data generating process is anticipated by
treating each new point as an indication to a potential new cluster.

References
----------
[1] Guedalia, I. D., London, M. and Werman, M.: An on-line agglomerative
    clustering method for nonstationary data, Neural Computation 11 (1999),
    521–540.
[2] Zhang, D., Chen, S. and Tan, K.: Improving the robustness of 'on-line
    agglomerative clustering method' based on kernel-induce distance measures.
    Neural Processing Letters 21 (2005), 45–51.
[3] Eppstein, D.: Fast hierarchical clustering and other applications of
    dynamic closest pairs. Journal of Experimental Algorithmics 5 (2000) 1.
[4] Souza, C. R.: Kernel functions for machine learning applications. Web.
    Last accessed June 28 (2016). <http://crsouza.blogspot.com/2010/03/kernel-
    functions-for-machine-learning.html>.
[5] Ben-Haim, Y. and Tom-Tov, E.: A streaming parallel decision tree algorithm.
    Journal of Machine Learning Research 11 (2010) 849-872.
"""

# Copyright (c) 2016, Carson J. Q. Farmer <carsonfarmer@gmail.com>
# Licensed under the MIT Licence (http://opensource.org/licenses/MIT).

from __future__ import print_function, division, absolute_import
from .kernel import kernel_dist, gaussian
from .centroid import Centroid, KernelCentroid
from operator import itemgetter
from fastpair import FastPair

# Idea: use the kernel induced distance to compute a 'weighted' convex hull of points seen so far.
# This might allow for non-circular clusters, and may in fact produce a more accurate clustering
# If we can, we might even be able to estiamte an α paramter for 'online' α-shapes!?


class AddC(object):
    """Implements the AddC clustering algorithm.

    For each data point arriving, the closest centroid to the incoming point
    is moved towards the point. If there are more than `kmax` centroids,
    then the two closest centroids are merged. This results in the creation of
    a redundant centroid; the redundant centroid is then set equal to the new
    data point. At any time, the data-structure can be queried for the current
    set of centroids/clusters, or updated with additional data points.
    """
    def __init__(self, kmax=100, dist=kernel_dist(gaussian),
                 centroid_factory=KernelCentroid):
        """Initialize an empty FastPair data-structure.

        Parameters
        ----------
        kmax : int, default=100
            The maximum number of cluster centroids to store (i.e., size of
            memory). This parameter controls the 'scale' of the desired
            solution, such that larger values of `kmax` will lead to a higher
            resolution cluster solution.
        """
        self.kmax = kmax
        self.npoints = 0
        self.centroid_factory = centroid_factory
        self.fastpair = FastPair(10, dist=dist)

    def __add__(self, p):
        """Add a point to the AddC sketch."""
        c = self.centroid_factory(p)  # Create an 'empty' centroid at point `p`
        self._step_one(c)
        self._step_two()
        self._step_three(c)
        self.npoints += 1  # Update count of points seen so far
        return self

    def __len__(self):
        """Number of points in the AddC sketch."""
        return len(self.fastpair)

    def __call__(self):
        """Return the current set of cluster centroids."""
        return self.centroids

    def __contains__(self, p):
        """Test if a given cluster centroid is in the AddC sketch."""
        return p in self.fastpair

    def __iter__(self):
        return iter(self.fastpair)

    def _step_one(self, c):
        # Step 1: Move the closest centroid towards the point
        if len(self.fastpair) > 0:
            # Single pass through list of neighbor points... this could also
            # be sped up with a spatial index, though harder to do with
            # kernel-induced distances
            # Alternatively, if it was possible to insert the new data point
            # _before_ querying for the closest pair (`step_two`), then we
            # could do it that way...
            old = min(self.fastpair.sdist(c), key=itemgetter(0))[1]
            self.fastpair._update_point(old, old.add(c))

    def _step_two(self):
        # Step 2: Merge the two closest centroids
        if len(self.fastpair) >= self.kmax and len(self.fastpair) > 1:
            dist, (a, b) = self.fastpair.closest_pair()
            self.fastpair -= b
            self.fastpair._update_point(a, a.merge(b))  # Update point `a`

    def _step_three(self, c):
        # Step 3: Set redundant centroid equal to new point
        self.fastpair += c

    def batch(self, points):
        # No checks, no nothing... just batch processing, pure and simple
        for point in points:
            self += point
        return self

    def trim(self, p=0.01):
        """Return only clusters over threshold."""
        sub = [x.size for x in self if x.size > 0]
        t = (sum(sub)/len(sub)) * p
        return [x for x in self if x.size >= t]

    @property
    def centroids(self):
        """For plotting."""
        return [c.center for c in self.fastpair]
