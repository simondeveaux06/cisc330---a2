"""
simulation.py — generate and evaluate tool-calibration observations.

Inputs (from the assignment's ground-truth model):
  M_TOOL : Figure 2 marker positions converted from cm to mm
  P_TOOL : known tip position, used to generate fixed-tip motions
  V_TOOL : known unit axis, used to evaluate the calibration error
Outputs:
  results/calibration_results.json : exact and noisy calibration results

Method:
  1. Generate fixed-tip pivot and fixed-shaft spin observations.
  2. Calibrate the tip and axis from exact marker observations.
  3. Choose Gaussian coordinate noise with 95% of 3-D errors below 1 mm.
  4. Add fresh marker noise and repeat the calibration for each trial.
  5. Compare the estimates with ground truth and summarize the errors.

Ground truth generates observations and checks results; it is not passed
into the calibration estimators. Run: python simulation.py
"""

import json
from math import erf, exp, pi, sqrt
from pathlib import Path

import numpy as np

from calibration import calibrate_axis, calibrate_tip, calibration_to_model
from geometry import (
    frame_from_points,
    invert_F,
    rot_x,
    rot_y,
    rot_z,
    transform_point,
    transform_vector,
)
from registration import CM_TO_MM, M_TOOL, P_TOOL, V_TOOL


def marker_sigma(radius=1.0, probability=0.95):
    """
    Find coordinate noise sigma for a specified 3-D error radius (mm).

    For isotropic Gaussian coordinates, error magnitude / sigma follows
    a chi distribution with three degrees of freedom. Invert its CDF
    by bisection, rather than using a one-dimensional two-sigma rule.
    """
    # Check the requested radius and probability.
    if radius <= 0 or not 0 < probability < 1:
        raise ValueError("need positive radius and probability in (0, 1)")

    # Find the dimensionless radius containing the desired probability.
    lower_bound = 0.0
    upper_bound = 20.0
    for _ in range(80):
        scaled_radius = (lower_bound + upper_bound) / 2
        cdf = (
            erf(scaled_radius / sqrt(2))
            - sqrt(2 / pi)
            * scaled_radius
            * exp(-scaled_radius * scaled_radius / 2)
        )
        if cdf < probability:
            lower_bound = scaled_radius
        else:
            upper_bound = scaled_radius

    # Convert the dimensionless radius to coordinate sigma in mm.
    quantile = (lower_bound + upper_bound) / 2
    return radius / quantile


def pivot_observations(markers=M_TOOL, tip=P_TOOL):
    """
    Generate 120 fixed-tip poses with varied tilt, azimuth, and roll.

    Sample 40 azimuths at each of 15, 35, and 60 degrees of tilt. Set
    t = pivot - R @ tip so every pose shares the same tracker-space tip.
    Return marker observations as an (n, 3, 3) array in mm.
    """
    # Place the pivot and prepare the ground-truth marker coordinates.
    pivot = np.array([30.0, -20.0, 80.0])
    markers = np.asarray(markers)
    observations = []

    # Vary multiple rotation axes while keeping the tip fixed.
    for tilt in (15, 35, 60):
        for azimuth in np.linspace(0, 360, 40, endpoint=False):
            rotation = rot_y(azimuth) @ rot_x(tilt) @ rot_y(2 * azimuth)
            translation = pivot - rotation @ tip
            observed_markers = markers @ rotation.T + translation
            observations.append(observed_markers)
    return np.array(observations)


def spin_observations(markers=M_TOOL, tip=P_TOOL):
    """
    Generate 120 full-circle spin poses, with no sliding or wobble.

    The illustrated model's shaft is parallel to y and passes through
    its tip. Rotate about y, then tilt the whole setup in tracker space.
    Return marker observations as an (n, 3, 3) array in mm.
    """
    # Place a tilted shaft at a fixed tracker-space tip position.
    shaft_orientation = rot_z(25) @ rot_x(20)
    fixed_tip = np.array([10.0, 40.0, 60.0])
    markers = np.asarray(markers)
    observations = []

    # Spin once around the shaft, excluding a duplicate 360-degree pose.
    for angle in np.linspace(0, 360, 120, endpoint=False):
        rotation = shaft_orientation @ rot_y(angle)
        translation = fixed_tip - rotation @ tip
        observed_markers = markers @ rotation.T + translation
        observations.append(observed_markers)
    return np.array(observations)


def angle_error(v, truth):
    """Return angular error in degrees, retaining sign errors."""
    # Use both sine and cosine terms to retain the full 0-180 deg range.
    v = np.asarray(v)
    truth = np.asarray(truth)
    cross_magnitude = np.linalg.norm(np.cross(v, truth))
    dot_product = np.dot(v, truth)
    return np.degrees(np.arctan2(cross_magnitude, dot_product))


def estimate(pivot, spin):
    """Calibrate from pivot and spin data; return both fits."""
    # Estimate the tip first; use it to choose the axis's forward sign.
    p_marker, pivot_tracker, pivot_rms = calibrate_tip(pivot)
    v_marker, plane_rms = calibrate_axis(spin, p_marker)
    return p_marker, v_marker, pivot_tracker, pivot_rms, plane_rms


def summarize(values):
    """Return trial error statistics, including SD and percentiles."""
    # Keep all summaries as plain floats for JSON output.
    values = np.asarray(values)
    return {
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "rmse": float(np.sqrt(np.mean(values ** 2))),
    }


def run(trials=500, seed=330):
    """Return exact recovery and noisy repeatability results."""
    # Generate exact observations and express truth in the marker frame.
    pivot_data = pivot_observations()
    spin_data = spin_observations()
    marker_from_model = invert_F(frame_from_points(*M_TOOL))
    p_true = transform_point(marker_from_model, P_TOOL)
    v_true = transform_vector(marker_from_model, V_TOOL)
    sigma = marker_sigma()

    # Recover the exact tip and axis, then convert to model coordinates.
    p_marker, v_marker, pivot_tracker, pivot_rms, plane_rms = estimate(
        pivot_data, spin_data
    )
    p_model, v_model = calibration_to_model(M_TOOL, p_marker, v_marker)

    # Record the acquisition settings and exact-data errors.
    result = {
        "length_unit": "mm",
        "figure_2_length_unit": "cm",
        "figure_2_to_mm_scale": CM_TO_MM,
        "seed": seed,
        "trials": trials,
        "pivot_poses": len(pivot_data),
        "spin_poses": len(spin_data),
        "sigma_per_coordinate_mm": sigma,
        "perfect": {
            "p_marker_mm": p_marker.tolist(),
            "v_marker": v_marker.tolist(),
            "p_model_mm": p_model.tolist(),
            "v_model": v_model.tolist(),
            "pivot_tracker_mm": pivot_tracker.tolist(),
            "tip_error_mm": float(np.linalg.norm(p_marker - p_true)),
            "axis_error_deg": float(angle_error(v_marker, v_true)),
            "pivot_rms_mm": float(pivot_rms),
            "plane_rms_mm": float(plane_rms),
        },
    }

    # Prepare reproducible, independent noise for each trial.
    rng = np.random.default_rng(seed)
    errors = []
    tip_estimates = []
    axis_estimates = []
    marker_error_norms = []
    for _ in range(trials):
        # Perturb marker coordinates before reconstructing any frames.
        pivot_noise = rng.normal(0, sigma, pivot_data.shape)
        spin_noise = rng.normal(0, sigma, spin_data.shape)
        p_marker, v_marker, _, pivot_rms, plane_rms = estimate(
            pivot_data + pivot_noise, spin_data + spin_noise
        )

        # Store calibration errors and estimates for bias analysis.
        errors.append([
            np.linalg.norm(p_marker - p_true),
            angle_error(v_marker, v_true),
            pivot_rms,
            plane_rms,
        ])
        tip_estimates.append(p_marker)
        axis_estimates.append(v_marker)

        # Track displacement magnitudes to check the 1 mm noise target.
        marker_error_norms.extend(
            np.linalg.norm(pivot_noise, axis=2).ravel()
        )
        marker_error_norms.extend(
            np.linalg.norm(spin_noise, axis=2).ravel()
        )

    # Summarize each error metric across complete calibration trials.
    errors = np.array(errors)
    marker_error_norms = np.array(marker_error_norms)
    metric_names = [
        "tip_error_mm",
        "axis_error_deg",
        "pivot_rms_mm",
        "plane_rms_mm",
    ]
    result["noisy"] = {}
    for column, name in enumerate(metric_names):
        result["noisy"][name] = summarize(errors[:, column])

    # Include mean estimates and the observed marker-noise distribution.
    result["noisy"].update({
        "mean_p_marker_mm": np.mean(tip_estimates, axis=0).tolist(),
        "mean_v_marker": np.mean(axis_estimates, axis=0).tolist(),
        "marker_error_samples": len(marker_error_norms),
        "marker_fraction_below_1mm": float(
            np.mean(marker_error_norms < 1.0)
        ),
        "marker_error_p95_mm": float(
            np.quantile(marker_error_norms, 0.95)
        ),
    })
    return result


def print_results(result):
    """Display screenshot-friendly exact and noisy calibration results."""
    # State the physical scale and the acquisition settings.
    print("\n=== Tool Calibration: setup ===")
    print("  Figure 2: cm converted to mm (scale = 10)")
    print(f"  pivot poses: {result['pivot_poses']}")
    print(f"  spin poses:  {result['spin_poses']}")
    perfect = result["perfect"]
    marker_from_model = invert_F(frame_from_points(*M_TOOL))
    comparisons = [
        ("tip (marker, mm)", transform_point(marker_from_model, P_TOOL),
         perfect["p_marker_mm"]),
        ("axis (marker)", transform_vector(marker_from_model, V_TOOL),
         perfect["v_marker"]),
        ("tip (model, mm)", P_TOOL, perfect["p_model_mm"]),
        ("axis (model)", V_TOOL, perfect["v_model"]),
        ("pivot (tracker, mm)", [30, -20, 80],
         perfect["pivot_tracker_mm"]),
    ]

    # Compare ground truth with exact calibration outputs.
    print("\n=== Test 1: exact-data calibration ===")
    for label, expected, computed in comparisons:
        expected = np.array2string(np.asarray(expected), precision=4,
                                   suppress_small=True)
        computed = np.array2string(np.round(computed, 4) + 0.0,
                                   precision=4, suppress_small=True)
        print(f"  {label}")
        print(f"    expected = {expected}, computed = {computed}")
    for label, key in [("tip error (mm)", "tip_error_mm"),
                       ("axis error (deg)", "axis_error_deg"),
                       ("pivot RMS (mm)", "pivot_rms_mm"),
                       ("plane RMS (mm)", "plane_rms_mm")]:
        print(f"  {label:<22} {perfect[key]:.3e}")
    exact_passed = (perfect["tip_error_mm"] < 1e-9
                    and perfect["axis_error_deg"] < 1e-9)
    print("  PASS" if exact_passed else "  FAIL")

    # Report measured noisy errors without inventing accuracy thresholds.
    noisy = result["noisy"]
    print("\n=== Noisy calibration: repeated trials ===")
    print(f"  trials: {result['trials']}, seed: {result['seed']}")
    print(f"  {'Error metric':<22} {'Mean':>9} {'SD':>9} "
          f"{'Median':>9} {'P95':>9} {'RMSE':>9}")
    for label, key in [("Tip position (mm)", "tip_error_mm"),
                       ("Axis direction (deg)", "axis_error_deg")]:
        values = " ".join(f"{noisy[key][stat]:9.4f}"
                          for stat in ["mean", "sd", "median", "p95",
                                       "rmse"])
        print(f"  {label:<22} {values}")

    # Show whether the observed marker noise agrees with the 95% target.
    print("\n=== Gaussian marker noise ===")
    print(f"  coordinate sigma (mm): "
          f"{result['sigma_per_coordinate_mm']:.8f}")
    print(f"  marker observations: {noisy['marker_error_samples']:,}")
    print("  required probability below 1 mm: 95%")
    print(f"  observed fraction below 1 mm: "
          f"{noisy['marker_fraction_below_1mm']:.4%}")
    print(f"  marker error P95 (mm): "
          f"{noisy['marker_error_p95_mm']:.6f}")
    print("  Vectors are rounded; errors use full precision.")


if __name__ == "__main__":
    # Run the experiment and save results beside this module.
    result = run()
    output_directory = Path(__file__).resolve().parent / "results"
    output_directory.mkdir(exist_ok=True)
    output_file = output_directory / "calibration_results.json"
    output_text = json.dumps(result, indent=2)
    output_file.write_text(output_text + "\n")
    print_results(result)
    print(f"\nFull numerical results saved to {output_file}")
