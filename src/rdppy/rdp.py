import numpy as np


def dist2(points, start, end):
    if np.allclose(start, end):
        return np.sum((points - start) ** 2)

    d = np.divide(end - start, np.sqrt(np.sum((end - start) ** 2)))

    max_p1 = np.dot(start - points, d).max()
    max_p2 = np.dot(points - end, d).max()

    return max([max_p1, max_p2, 0]) ** 2 + np.cross(points - start, d) ** 2


def _lossy_compress(points, threshold):
    start = points[0]
    end = points[-1]

    d = dist2(points, start, end)
    i = np.argmax(d)
    return (
        np.vstack(
            (
                _lossy_compress(points[: i + 1], threshold)[:-1],
                _lossy_compress(points[i:], threshold),
            )
        )
        if d[i] > threshold ** 2
        else np.array([start, end])
    )


def lossy_compress(points, threshold):
    points = np.array(points)
    if len(points.shape) != 2 or points.shape[-1] not in (2, 3):
        raise ValueError("Points have to be a sequence of 2D or 3D tuple.")

    return _lossy_compress(points, threshold)
