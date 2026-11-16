"""
fitting.py — shared plane-fitting utility for tool-axis calibration.

Input:
  points : observed 3-D positions from one marker's spin trajectory
Outputs:
  center : mean position of the observed points
  normal : unit normal to the fitted plane (either sign)
  RMS    : perpendicular fit residual in the input coordinate units

Method:
  1. Subtract the mean to center the point cloud.
  2. Use SVD to find the direction of least variation.
  3. Use that direction as the plane normal and measure the residual.
"""

import numpy as np


def fit_plane(points):
    """
    Fit a plane to at least three non-collinear points by least squares.

    Parameters
    ----------
    points : (n, 3) observed coordinates

    Returns
    -------
    center : (3,) mean point, lying on the fitted plane
    normal : (3,) unit plane normal, with undetermined sign
    rms    : scalar RMS perpendicular distance to the plane
    """
    # Check that the point cloud has enough finite 3-D observations.
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError("points must be (n, 3), with n >= 3")
    if not np.all(np.isfinite(points)):
        raise ValueError("points must be finite")

    # Center the points and find their principal directions.
    center = points.mean(axis=0)
    centered_points = points - center
    _, singular_values, principal_directions = np.linalg.svd(
        centered_points, full_matrices=False
    )

    # Reject clouds that span only a point or a line.
    tolerance = 1e-10 * max(singular_values[0], 1.0)
    if singular_values[1] <= tolerance:
        raise ValueError("points are stationary or collinear; no unique plane")

    # The least-varying direction is perpendicular to the fitted plane.
    normal = principal_directions[-1]
    distances = centered_points @ normal
    rms = np.sqrt(np.mean(distances ** 2))
    return center, normal, rms
