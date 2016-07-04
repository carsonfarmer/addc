#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""AddC: Data-structure for online/streaming clustering.

Testing module for AddC.
"""

# Copyright (c) 2016, Carson J. Q. Farmer <carsonfarmer@gmail.com>
# Licensed under the MIT Licence (http://opensource.org/licenses/MIT).

from __future__ import print_function, division, absolute_import

# from operator import itemgetter
# from types import FunctionType
# from itertools import cycle, combinations, groupby
import random
import pytest
from addc import AddC, Centroid, KernelCentroid
from math import isinf, isnan

def contains_same(s, t):
    s, t = set(s), set(t)
    return s >= t and s <= t


def all_close(s, t, tol=1e-8):
    # Ignores inf and nan values...
    return all(abs(a - b) < tol for a, b in zip(s, t)
               if not isinf(a) and not isinf(b) and
                  not isnan(a) and not isnan(b))

def rand_normal(center=(0, 0), sd=0.02):
    return tuple([random.normalvariate(m, sd) for m in center])


def rand_tuple(dim=2):
    return tuple([random.random() for _ in range(dim)])


@pytest.fixture(scope="module")
def ClusterSet(n=100, means=[(.6, .5), (.3, .8), (.2, .4)], sd=0.02):
    # random.seed(8714)
    data = [rand_normal(mean, sd) for _ in range(n) for mean in means]
    data += [rand_tuple() for _ in range(n//2)]  # Noise
    return data


@pytest.fixture(scope="module")
def PointSet(n=50, d=10):
    """Return numpy array of shape `n`x`d`."""
    # random.seed(8714)
    return [rand_tuple(d) for _ in range(n)]


class TestAddC:
    """Main test class."""

    def test_init(self):
        ac = AddC()
        assert len(ac) < 1
        # More asserts need to be added here

    def test_add(self):
        ps = PointSet()
        ac = AddC()
        for p in ps:
            ac += p
        assert len(ac) == len(ps)
        # More asserts need to be added here

    def test_batch(self):
        ps = PointSet()
        ac = AddC()
        assert len(ac) == 0
        ac.batch(ps)
        assert len(ac) == len(ps)

    def test_compare_batch_add(self):
        ps = PointSet()
        ac1 = AddC()
        ac2 = AddC()
        for p in ps:
            ac1 += p
        ac2.batch(ps)
        for a, b in zip(ac1.centroids, ac2.centroids):
            assert all_close(a, b)
        for a, b in zip(ac1, ac2):
            assert all_close(a.center, b.center)
            assert a.count == b.count
            assert abs(a.size - b.size) < 1e-8

    def test_kernel(self):
        # For now, we're only testing the gaussian kernel
        ps = PointSet()
        ac = AddC().batch(ps)
        # assert getattr(ac.dist, "__name__") == "gaussian"
        assert (ac.fastpair.dist((1, 2, 3, 4, 5), (6, 5, 4, 3, 2)) - 2) < 1e-8

    def test_len(self):
        ps = PointSet()
        ac = AddC(kmax=8)
        assert len(ac) == 0
        ac.batch(ps)
        assert len(ac) == 8

    def test_contains(self):
        ps = PointSet(10, 2)
        ac = AddC()
        assert ps[0] not in ac
        ac.batch(ps)
        assert ac.centroids[0] in ac

    def test_call_and_centroids(self):
        ps = PointSet()
        ac = AddC(kmax=5)
        assert len(ac()) == len(ac.centroids) == 0
        ac.batch(ps)
        assert len(ac()) == len(ac.centroids) == 5

    def test_iter(self):
        ps = PointSet()
        ac = AddC().batch(ps)
        assert hasattr(ac, '__iter__')
        assert len(ac) == len(ps)

    def test_npoints(self):
        ps = PointSet()
        ac = AddC(10).batch(ps)
        assert ac.npoints == len(ps)
        assert ac.npoints > len(ac)
        assert isinstance(ac.npoints, int)

    # def test_merge(self):
    #     # I'm not so sure this will work?
    #     ps = PointSet()
    #     n = len(ps)
    #     ac1 = AddC(20).batch(ps[:n//2])
    #     ac2 = AddC(20).batch(ps[n//2:])
    #     ac3 = AddC(20).batch(ps)
    #     ac1 = ac1.merge(ac2)
    #     assert ac3.npoints == ac1.npoints == len(ps)
    #     assert len(ac3) == len(ac1) == 20
    #     print(set(ac3.centroids).difference(ac1.centroids))
    #     assert contains_same(ac3.centroids, ac1.centroids)

    def test_cluster(self):
        means=[(.6, .5), (.3, .8), (.2, .4)]
        sd = 0.05
        ps = ClusterSet(means=means, sd=sd)
        ac = AddC(kmax=6).batch(ps)
        centroids = ac.trim(0.2)
        assert len(centroids) == len(means)
        for a, b in zip(sorted(centroids), sorted(means)):
            assert all_close(a, b, sd)  # Tolerance equal to sd...

class TestCentroid:
    def test_init(self):
        compare = (1, 2, 3, 4, 5)
        with pytest.raises(TypeError):
            c = KernelCentroid()
        c = KernelCentroid(compare)
        assert c.count == 0
        assert c.size == 0
        assert len(c.center) == len(c) == len(compare)
        assert c.center == compare

    def test_cast(self):
        compare = (1, 2, 3, 4, 5)
        c = Centroid(compare)
        assert tuple(c) == compare
        assert len(tuple(c)) == len(compare)
        for i, item in enumerate(c):
            assert item == compare[i]

    def test_mutate(self):
        c = Centroid((1, 2, 3, 4, 5))
        with pytest.raises(AttributeError):
            c.center = None
