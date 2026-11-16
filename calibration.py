"""
calibration.py — calibrate the tracked biopsy needle's tip and axis.

Inputs (from optical-tracker observations):
  pivot observations : three marker positions at each fixed-tip pose
  spin observations  : three marker positions at each shaft-spin pose
  Marker order is M1, M2, M3 in every pose; coordinates are in mm.
Outputs:
  p_marker : tool-tip position in the marker frame
  v_marker : unit tool-axis direction in the marker frame
  pivot    : fixed tip position in the tracker frame
  RMS      : residual distances for the pivot and plane fits (mm)

Method (Calibration slides 5-7 and 11):
  The tip and axis are rigidly fixed to the marker assembly, so their
  coordinates stay constant in the marker frame as the tool moves.
  1. Build a marker-to-tracker frame from each observed triangle.
     Use the same centroid and axis convention as registration.py.
  2. Tip: keep the tip at a fixed pivot and solve all pose equations:
         R_i @ p_marker + t_i = pivot
     Stack [R_i | -I] and solve for p_marker and pivot by least squares.
  3. Axis: spin the shaft without sliding or tilting. Each marker traces
     a circle; fit its plane and average the aligned plane normals.
  4. Rotate the tracker-space axis back into each marker frame, then
     average and normalize. Translation does not affect a direction.
  5. Choose the sign toward the calibrated tip. Assignment Figure 2
     places the marker centroid behind the tip along the shaft.
  6. Convert the calibrated point and direction into the illustrated
     model frame before using them with M_TOOL in register_tool.

Ground-truth tip, axis, and poses are not inputs to these estimators.
"""

import numpy as np

from fitting import fit_plane
from geometry import frame_from_points, transform_point, transform_vector


def tracked_frames(observations):
    """
    Build marker-to-tracker transforms from observed marker coordinates.

    Parameters
    ----------
    observations : (n, 3, 3) marker coordinates, ordered M1, M2, M3

    Returns
    -------
    frames : (n, 4, 4) rigid transforms from marker to tracker frame
    """
    # Check the pose count, marker layout, and coordinate values.
    observations = np.asarray(observations, dtype=float)
    if observations.ndim != 3 or observations.shape[1:] != (3, 3):
        raise ValueError("observations must be (n, 3, 3)")
    if len(observations) < 3 or not np.all(np.isfinite(observations)):
        raise ValueError("need at least three finite tracked poses")

    # Use the existing registration convention for each marker frame.
    frames = []
    for markers in observations:
        frames.append(frame_from_points(*markers))
    return np.array(frames)


def calibrate_tip(observations):
    """
    Recover the tip from fixed-tip pivot observations (slides 6-7).

    Parameters
    ----------
    observations : (n, 3, 3) marker coordinates during pivoting

    Returns
    -------
    p_marker  : (3,) tool tip in the marker frame (mm)
    pivot     : (3,) fixed tip position in the tracker frame (mm)
    pivot_rms : scalar RMS distance from the fitted pivot (mm)
    """
    # Build one equation block: R_i @ p_marker - pivot = -t_i.
    frames = tracked_frames(observations)
    equation_blocks = []
    for frame in frames:
        rotation = frame[:3, :3]
        equation_blocks.append(np.column_stack([rotation, -np.eye(3)]))

    # Stack all poses and solve for the six unknown coordinates.
    coefficients = np.vstack(equation_blocks)
    targets = -frames[:, :3, 3].reshape(-1)
    solution, _, rank, _ = np.linalg.lstsq(
        coefficients, targets, rcond=None
    )
    if rank < 6:
        raise ValueError(
            "pivot poses do not sufficiently excite multiple rotation axes"
        )

    # Measure how closely the transformed tips share one pivot.
    predicted = np.sum(coefficients * solution, axis=1)
    residuals = (predicted - targets).reshape(-1, 3)
    pivot_rms = np.sqrt(np.mean(np.sum(residuals ** 2, axis=1)))
    p_marker = solution[:3]
    pivot = solution[3:]
    return p_marker, pivot, pivot_rms


def calibrate_axis(observations, p_marker):
    """
    Recover the axis from fixed-shaft spin observations (slide 11).

    Parameters
    ----------
    observations : (n, 3, 3) marker coordinates during pure shaft spin
    p_marker     : (3,) calibrated tip in the marker frame (mm)

    Returns
    -------
    v_marker  : (3,) unit tipward shaft direction in the marker frame
    plane_rms : scalar pooled perpendicular plane-fit RMS (mm)

    Keep the shaft line and axial position fixed during spin. The sign
    rule requires the marker centroid to lie behind the tip along the
    shaft, as in Figure 2. A different tool may need another sign cue.
    """
    # Build the tracked frames and check the calibrated tip.
    observations = np.asarray(observations, dtype=float)
    frames = tracked_frames(observations)
    p_marker = np.asarray(p_marker, dtype=float)
    if p_marker.shape != (3,) or not np.all(np.isfinite(p_marker)):
        raise ValueError("p_marker must be a finite 3-vector")

    # Fit each marker's circle plane and align its normal's sign.
    normals = []
    plane_residuals = []
    for marker_index in range(3):
        trajectory = observations[:, marker_index, :]
        _, normal, rms = fit_plane(trajectory)
        if normals and np.dot(normal, normals[0]) < 0:
            normal = -normal
        normals.append(normal)
        plane_residuals.append(rms)

    # Average the aligned normals to estimate the tracker-space axis.
    v_tracker = np.mean(normals, axis=0)
    v_tracker /= np.linalg.norm(v_tracker)

    # Rotate the direction back into each marker frame, then average.
    marker_directions = []
    for frame in frames:
        rotation = frame[:3, :3]
        marker_directions.append(rotation.T @ v_tracker)
    v_marker = np.mean(marker_directions, axis=0)
    v_marker /= np.linalg.norm(v_marker)

    # Choose the direction toward the tip, using Figure 2's geometry.
    projection = np.dot(v_marker, p_marker)
    sign_tolerance = 1e-8 * max(np.linalg.norm(p_marker), 1.0)
    if abs(projection) <= sign_tolerance:
        raise ValueError(
            "tipward sign is undefined: use a different physical convention"
        )
    if projection < 0:
        v_marker = -v_marker

    # Pool the three plane residuals into one RMS distance.
    plane_rms = np.sqrt(np.mean(np.square(plane_residuals)))
    return v_marker, plane_rms


def calibration_to_model(markers_model, p_marker, v_marker):
    """
    Convert calibrated tip and axis for use with register_tool.

    Parameters
    ----------
    markers_model : (3, 3) marker coordinates in the model frame
    p_marker      : (3,) calibrated tip in the marker frame (mm)
    v_marker      : (3,) calibrated unit axis in the marker frame

    Returns
    -------
    p_model : (3,) tip in the model frame (mm)
    v_model : (3,) unit axis in the model frame
    """
    # Locate the marker frame within the illustrated model frame.
    markers_model = np.asarray(markers_model, dtype=float)
    model_from_marker = frame_from_points(*markers_model)

    # Transform the tip as a point and the axis as a direction.
    p_model = transform_point(model_from_marker, p_marker)
    v_model = transform_vector(model_from_marker, v_marker)
    return p_model, v_model
