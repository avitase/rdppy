# RDPpy

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyPI](https://img.shields.io/pypi/v/rdppy)](https://pypi.org/project/rdppy/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Implementation of the Ramer-Douglas-Peucker algorithm for polylines using a point-to-line-segment distance measure.

## Usage
The API consists of a single function `rdppy.filter(points, threshold)` which returns a binary mask upon calling with a sequence of points and a threshold (aka the _epsilon_ value):
```python
>>> import rdppy
>>> points = [(0, 0),
...           (4, 0),
...           (0, 1),
...           (1, 1),
...           (1, 2),
...           (2, 2),
...           (2, 3),
...           (3, 3),
...           (3, 4),
...           (5, 4)]
>>> mask = rdppy.filter(points, threshold=.9)
>>> mask
array([True, True, True, False, False, False, False, False, True, True])
```
This mask is a `numpy array` and can be used to filter a given sequence, e.g.,
```python
>>> import numpy as np
>>> np.array(points)[mask]
array([[0, 0],
       [4, 0],
       [0, 1],
       [3, 4],
       [5, 4]])
```
Note that this allows the filtering of more complex sequences which carry, for instance, meta information:
```python
>>> points = np.array([(0, 0, 'a'),
...                    (4, 0, 'b'),
...                    (0, 1, 'c'),
...                    (1, 1, 'd'),
...                    (1, 2, 'e'),
...                    (2, 2, 'f'),
...                    (2, 3, 'g'),
...                    (3, 3, 'h'),
...                    (3, 4, 'i'),
...                    (5, 4, 'j')])
>>> mask = rdppy.filter([(float(x), float(y)) for x, y, _ in points], .9)
>>> points[mask, -1]
array(['a', 'b', 'c', 'i', 'j'], dtype='<U21')
```
**The default metric only works for 2D points** but users may define custom metrics to measure the distance between a list of points, `points`, of any dimension and a segment parametrized via its start, `seg_start`, and end `seg_end`. For instance `my_dist2` measures the distances of 2D points to the (infinite) line rather than the finite segment:
```python
>>> def my_dist2(points, seg_start, seg_end):
...    d = np.divide(seg_end - seg_start, np.sqrt(np.sum((seg_end - seg_start) ** 2)))
...    return np.cross(points - np.expand_dims(seg_start, 0), np.expand_dims(d, 0)) ** 2
    
>>> rdppy.filter(points, threshold=.9, dist2_fun=my_dist2)
```
The maximum of the returned values is compared with the squared threshold value. By default the function `rdp.dist2` is used:
```python
rdppy.filter(points, threshold, dist2_fun=rdppy.rdp.dist2)
```