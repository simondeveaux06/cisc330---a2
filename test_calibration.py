"""
test_calibration.py — check tool calibration against known results.

Checks:
  1. Exact recovery and integration with the existing registration.
  2. Invariance when tracker coordinates or marker-frame axes change.
  3. Recovery when the model-frame tip is not at the origin.
  4. Rejection of motions that cannot determine the calibration.
  5. Gaussian noise scale and recovery from noisy observations.

Run: python test_calibration.py
"""

import numpy as np

from calibration import calibrate_axis, calibrate_tip, calibration_to_model
from fitting import fit_plane
from geometry import (
    frame_from_points,
    invert_F,
    make_F,
    rot_x,
    rot_z,
    transform_point,
    transform_vector,
)
from registration import M_TOOL, P_TOOL, V_TOOL, register_tool
from simulation import (
    angle_error,
    marker_sigma,
    pivot_observations,
    spin_observations,
)



def check_vector(output, frame, expected, computed, direction=False,
                 tolerance=1e-9):
    """Print one comparison and return its row for the results table."""
    # Round vector displays while retaining full precision for errors.
    expected = np.asarray(expected, dtype=float)
    computed = np.asarray(computed, dtype=float)
    expected_text = np.array2string(
        expected, precision=4, suppress_small=True
    )
    computed_text = np.array2string(
        np.round(computed, 4) + 0.0, precision=4, suppress_small=True
    )
    error = (angle_error(computed, expected) if direction else
             np.linalg.norm(computed - expected))
    unit = "deg" if direction else "mm"
    passed = error < tolerance
    print(f"  expected {output} = {expected_text}")
    print(f"  computed {output} = {computed_text} ({frame})")
    print(f"  {output} error ({unit}): {error:.3e}")
    return (output, frame, expected_text, computed_text,
            f"{error:.3e}", unit, "PASS" if passed else "FAIL")


def test_perfect_and_registration():
    """Check exact recovery and tool-to-CT registration."""
    # Calibrate from exact observations and compare with known values.
    p_marker, pivot, pivot_rms = calibrate_tip(pivot_observations())
    v_marker, plane_rms = calibrate_axis(spin_observations(), p_marker)
    assert np.allclose(p_marker, [-40, -200, 0], atol=1e-10)
    assert np.allclose(pivot, [30, -20, 80], atol=1e-10)
    assert angle_error(v_marker, V_TOOL) < 1e-10
    assert pivot_rms < 1e-10 and plane_rms < 1e-10

    # Put the illustrated model into a known CT pose.
    p_model, v_model = calibration_to_model(M_TOOL, p_marker, v_marker)
    ct_from_model = make_F(rot_z(35) @ rot_x(20), [100, -50, 200])
    observed_markers = np.array([
        transform_point(ct_from_model, marker) for marker in M_TOOL
    ])

    # Compare registration outputs with the directly transformed truth.
    p_ct, v_ct, _ = register_tool(
        M_TOOL, p_model, v_model, observed_markers
    )
    expected_tip = transform_point(ct_from_model, P_TOOL)
    expected_axis = transform_vector(ct_from_model, V_TOOL)
    assert np.allclose(p_ct, expected_tip, atol=1e-10)
    assert np.allclose(v_ct, expected_axis, atol=1e-10)


    # Show exact calibration and registration comparisons.
    print(f"  pivot residual RMS (mm): {pivot_rms:.3e}")
    print(f"  plane residual RMS (mm): {plane_rms:.3e}")
    return [
        check_vector("Tip", "marker", [-40, -200, 0], p_marker),
        check_vector("Axis", "marker", V_TOOL, v_marker, direction=True),
        check_vector("Pivot", "tracker", [30, -20, 80], pivot),
        check_vector("Tip", "model", P_TOOL, p_model),
        check_vector("Tip", "CT", expected_tip, p_ct),
        check_vector("Axis", "CT", expected_axis, v_ct, direction=True),
    ]


def test_changed_tracker_frame():
    """Check calibration after changing tracker coordinates."""
    # Rotate and translate both acquisition sets in tracker space.
    tracker_change = make_F(rot_x(37) @ rot_z(-24), [-110, 25, 300])
    rotation = tracker_change[:3, :3]
    translation = tracker_change[:3, 3]
    pivot_data = pivot_observations() @ rotation.T + translation
    spin_data = spin_observations() @ rotation.T + translation

    # Intrinsic parameters stay fixed; the tracker-space pivot moves.
    p_marker, pivot, _ = calibrate_tip(pivot_data)
    v_marker, _ = calibrate_axis(spin_data, p_marker)
    expected_pivot = transform_point(tracker_change, [30, -20, 80])
    assert np.allclose(p_marker, [-40, -200, 0], atol=1e-10)
    assert np.allclose(pivot, expected_pivot, atol=1e-10)
    assert angle_error(v_marker, V_TOOL) < 1e-10


    # Confirm intrinsic coordinates survive a tracker-frame change.
    return [
        check_vector("Tip", "marker", [-40, -200, 0], p_marker),
        check_vector("Axis", "marker", V_TOOL, v_marker, direction=True),
        check_vector("Pivot", "tracker", expected_pivot, pivot),
    ]


def test_reordered_marker_frame():
    """Check recovery after cyclically reordering the markers."""
    # Reorder all observations to change the marker-frame axes.
    order = [1, 2, 0]
    markers_model = M_TOOL[order]
    pivot_data = pivot_observations()[:, order]
    spin_data = spin_observations()[:, order]
    p_marker, _, _ = calibrate_tip(pivot_data)
    v_marker, _ = calibrate_axis(spin_data, p_marker)

    # Compare against truth expressed in the newly defined marker frame.
    marker_from_model = invert_F(frame_from_points(*markers_model))
    expected_tip = transform_point(marker_from_model, P_TOOL)
    expected_axis = transform_vector(marker_from_model, V_TOOL)
    assert np.allclose(p_marker, expected_tip, atol=1e-10)
    assert angle_error(v_marker, expected_axis) < 1e-10

    # Convert back to model coordinates and check the original truth.
    p_model, v_model = calibration_to_model(
        markers_model, p_marker, v_marker
    )
    assert np.allclose(p_model, P_TOOL, atol=1e-10)
    assert angle_error(v_model, V_TOOL) < 1e-10


    # Show recovery in the reordered marker frame.
    return [
        check_vector("Tip", "marker", expected_tip, p_marker),
        check_vector("Axis", "marker", expected_axis, v_marker,
                     direction=True),
    ]


def test_nonzero_model_tip():
    """Check recovery when the model tip is away from the origin."""
    # Shift model coordinates, keeping the physical geometry fixed.
    shift = np.array([8.0, -3.0, 11.0])
    shifted_markers = M_TOOL + shift
    shifted_tip = P_TOOL + shift
    pivot_data = pivot_observations(shifted_markers, shifted_tip)
    spin_data = spin_observations(shifted_markers, shifted_tip)

    # Recover the shifted model tip and unchanged axis direction.
    p_marker, _, _ = calibrate_tip(pivot_data)
    v_marker, _ = calibrate_axis(spin_data, p_marker)
    p_model, v_model = calibration_to_model(
        shifted_markers, p_marker, v_marker
    )
    assert np.allclose(p_model, shifted_tip, atol=1e-10)
    assert angle_error(v_model, V_TOOL) < 1e-10


    # Show recovery of a tip away from the model origin.
    return [
        check_vector("Tip", "model", shifted_tip, p_model),
        check_vector("Axis", "model", V_TOOL, v_model, direction=True),
    ]


def test_degenerate_motion():
    """Reject undefined pivot, plane, and axis-sign estimates."""
    # Stationary poses and shaft spin cannot determine the full tip.
    stationary_data = np.repeat(M_TOOL[None], 4, axis=0)
    for observations in (stationary_data, spin_observations()):
        try:
            calibrate_tip(observations)
        except ValueError:
            pass
        else:
            raise AssertionError("unobservable pivot accepted")

    print("  stationary pivot and spin-only pivot: rejected as expected")

    # Collinear points do not define a unique plane normal.
    try:
        fit_plane([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    except ValueError:
        pass
    else:
        raise AssertionError("collinear plane fit accepted")

    print("  collinear plane: rejected as expected")

    # A tip at the marker origin cannot resolve the tipward axis sign.
    try:
        calibrate_axis(spin_observations(), [0, 0, 0])
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous direction sign accepted")


    print("  ambiguous tipward sign: rejected as expected")
    return [
        (case, "n/a", "Reject input", "Rejected", "n/a", "n/a", "PASS")
        for case in ["Stationary", "Spin-only", "Collinear", "Axis sign"]
    ]


def test_noise_scale_and_recovery():
    """Check the noise radius and one noisy calibration."""
    # About 95% of independent 3-D marker errors should be below 1 mm.
    rng = np.random.default_rng(17)
    sigma = marker_sigma()
    marker_errors = rng.normal(0, sigma, (100000, 3))
    fraction_below_1mm = np.mean(np.linalg.norm(marker_errors, axis=1) < 1)
    assert abs(fraction_below_1mm - 0.95) < 0.003

    # Add coordinate noise before running either calibration estimator.
    pivot_data = pivot_observations()
    spin_data = spin_observations()
    noisy_pivot = pivot_data + rng.normal(0, sigma, pivot_data.shape)
    noisy_spin = spin_data + rng.normal(0, sigma, spin_data.shape)
    p_marker, _, _ = calibrate_tip(noisy_pivot)
    v_marker, _ = calibrate_axis(noisy_spin, p_marker)

    # Check broad error limits for this fixed noise realization.
    assert np.linalg.norm(p_marker - [-40, -200, 0]) < 2.0
    assert angle_error(v_marker, V_TOOL) < 3.0

    # Display the measured noise coverage and noisy recovery limits.
    print(f"  coordinate sigma (mm): {sigma:.8f}")
    print(f"  marker errors below 1 mm: {fraction_below_1mm:.2%}")
    print("  noisy recovery limits: tip < 2 mm, axis < 3 deg")
    return [
        check_vector("Tip", "marker", [-40, -200, 0], p_marker,
                     tolerance=2.0),
        check_vector("Axis", "marker", V_TOOL, v_marker,
                     direction=True, tolerance=3.0),
    ]


if __name__ == "__main__":
    # Run each existing test group and print its outcome.
    tests = [
        test_perfect_and_registration,
        test_changed_tracker_frame,
        test_reordered_marker_frame,
        test_nonzero_model_tip,
        test_degenerate_motion,
        test_noise_scale_and_recovery,
    ]
    # Print each test's details and collect the comparison rows.
    names = ["Exact recovery and registration", "Changed tracker frame",
             "Reordered markers", "Nonzero model tip", "Degenerate motions",
             "Noisy recovery"]
    rows = []
    passed_count = 0
    for number, (name, test) in enumerate(zip(names, tests), start=1):
        print(f"\n=== Test {number}: {name} ===")
        try:
            comparisons = test()
        except AssertionError as error:
            print(f"  FAIL: {error}")
            continue
        rows.extend((str(number), *row) for row in comparisons)
        passed = all(row[-1] == "PASS" for row in comparisons)
        print("  PASS" if passed else "  FAIL")
        passed_count += int(passed)

    # Match the report table layout, with explicit frames and error units.
    print("\n=== Calibration test summary ===")
    headers = ("Test", "Output", "Frame", "Predicted", "Computed",
               "Error", "Unit", "Result")
    widths = [max(len(str(row[i])) for row in [headers, *rows])
              for i in range(len(headers))]
    for row in [headers, *rows]:
        print("  ".join(str(value).ljust(width)
                        for value, width in zip(row, widths)))
    print("\nVectors are rounded for display; errors use full precision.")
    print(f"{passed_count}/{len(tests)} tests passed")
    raise SystemExit(0 if passed_count == len(tests) else 1)
