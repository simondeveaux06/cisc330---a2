"""Independent checks for CISC 330 Targeting Error Simulation.

Run alongside targeting_error_simulation.py and the group's existing modules:
    python test_targeting_error_simulation.py
All physical lengths are in millimetres.
"""
import unittest
import numpy as np

from geometry import point_line_distance
from registration import M_TOOL, P_TOOL, V_TOOL, register_tool, CM_TO_MM
from simulation import marker_sigma
from targeting_error_simulation import (
    MARKER_SIGMA_MM, INSERTION_DEPTH_MM, ERROR_LIMIT_MM,
    make_trial_data, estimated_needle_poses, evaluate_sigma,
    wilson_interval,
)


class TargetingSimulationTests(unittest.TestCase):
    def test_perfect_data_baseline(self):
        """Exact marker observations must recover the true tool and hit target."""
        tip, direction, F = register_tool(M_TOOL, P_TOOL, V_TOOL, M_TOOL)
        target = P_TOOL + INSERTION_DEPTH_MM * V_TOOL
        np.testing.assert_allclose(tip, P_TOOL, atol=1e-10)
        np.testing.assert_allclose(direction, V_TOOL, atol=1e-10)
        np.testing.assert_allclose(F, np.eye(4), atol=1e-10)
        np.testing.assert_allclose(target, [0, -100, 0], atol=1e-10)
        self.assertAlmostEqual(point_line_distance(target, tip, direction), 0.0, places=9)
        result = evaluate_sigma(0, tip[None], direction[None], np.zeros((1, 3)))
        self.assertEqual(result["trajectory_success_rate"], 1.0)
        self.assertEqual(result["tip_success_rate"], 1.0)

    def test_marker_noise_distribution(self):
        """Check per-coordinate sigma and empirical 3-D radial 95% criterion."""
        self.assertAlmostEqual(MARKER_SIGMA_MM, marker_sigma(1, 0.95), places=12)
        rng = np.random.default_rng(3301)
        marker_noise, target_normals = make_trial_data(40000, rng)
        self.assertEqual(marker_noise.shape, (40000, 3, 3))
        radii = np.linalg.norm(marker_noise, axis=2).ravel()
        fraction = np.mean(radii < 1.0)
        print(f"  empirical P(marker 3D error < 1 mm) = {fraction:.4%}")
        self.assertLess(abs(fraction - .95), .003)
        self.assertLess(abs(marker_noise.mean()), .01)
        self.assertLess(abs(marker_noise.std() - MARKER_SIGMA_MM), .01)
        # Independence of x/y/z marker errors and independent target draws.
        corr = np.corrcoef(np.column_stack((
            marker_noise[:, 0, 0], marker_noise[:, 0, 1],
            marker_noise[:, 1, 0], target_normals[:, 0]
        )).T)
        self.assertLess(np.max(np.abs(corr - np.eye(4))), .025)

    def test_noise_propagates_through_registration(self):
        """Match simulation poses to independent direct registration calls."""
        rng = np.random.default_rng(3302)
        noise, normals = make_trial_data(120, rng)
        tips, directions = estimated_needle_poses(noise)
        for i in [0, 1, 20, 119]:
            expected_tip, expected_direction, _ = register_tool(
                M_TOOL, P_TOOL, V_TOOL, M_TOOL + noise[i]
            )
            np.testing.assert_allclose(tips[i], expected_tip, atol=1e-10)
            np.testing.assert_allclose(directions[i], expected_direction, atol=1e-10)
        self.assertTrue(np.allclose(np.linalg.norm(directions, axis=1), 1.0))
        self.assertGreater(np.std(tips[:, 0]), 0.01)
        self.assertGreater(np.std(directions[:, 0]), 0.001)
        # Explicit geometry check: one translated observation shifts the tip,
        # but does not rotate the axis.
        shift = np.array([1.2, -0.7, 0.4])
        shifted_tip, shifted_dir, _ = register_tool(
            M_TOOL, P_TOOL, V_TOOL, M_TOOL + shift
        )
        np.testing.assert_allclose(shifted_tip, P_TOOL + shift, atol=1e-10)
        np.testing.assert_allclose(shifted_dir, V_TOOL, atol=1e-10)

    def test_independent_target_and_trajectory_metric(self):
        """Target sigma changes target position, not re-registered needle."""
        rng = np.random.default_rng(3303)
        marker_noise, target_normals = make_trial_data(200, rng)
        tips, directions = estimated_needle_poses(marker_noise)
        a = evaluate_sigma(0.0, tips, directions, target_normals)
        b = evaluate_sigma(2.0, tips, directions, target_normals)
        self.assertEqual(a["trials"], b["trials"])
        self.assertNotEqual(a["trajectory_mean_mm"], b["trajectory_mean_mm"])
        # Point on infinite needle line has zero trajectory error, while
        # its fixed-depth tip distance can be nonzero.
        tip = np.array([0., 0., 0.])
        direction = np.array([0., -1., 0.])
        target = np.array([0., -102., 0.])
        self.assertAlmostEqual(point_line_distance(target, tip, direction), 0.)
        result = evaluate_sigma(0, tip[None], direction[None],
                                np.array([[0., -2., 0.]]))
        self.assertEqual(result["trajectory_success_rate"], 1.)
        # The secondary metric is defined at exactly 100 mm.
        self.assertAlmostEqual(np.linalg.norm(target - (tip + 100 * direction)), 2.)

    def test_units_and_thresholds(self):
        """Figure 2 is cm, but M_TOOL and all noise inputs are already mm."""
        self.assertEqual(CM_TO_MM, 10.)
        np.testing.assert_allclose(M_TOOL, [[20, 180, 0], [80, 180, 0], [20, 240, 0]])
        self.assertAlmostEqual(np.linalg.norm(M_TOOL[1] - M_TOOL[0]), 60.)
        self.assertEqual(INSERTION_DEPTH_MM, 100.)
        self.assertEqual(ERROR_LIMIT_MM, 3.)
        self.assertTrue(0 < MARKER_SIGMA_MM < 1.)

    def test_sweep_reproducibility_and_export_consistency(self):
        """Fixed seed reproduces sweep, and saved CSV/JSON agree with it."""
        from targeting_error_simulation import run_sweep, SEED
        rng1 = np.random.default_rng(SEED)
        rng2 = np.random.default_rng(SEED)
        a = run_sweep([0.0, 0.25], 40, rng1)
        b = run_sweep([0.0, 0.25], 40, rng2)
        self.assertEqual(a, b)
        from pathlib import Path
        import csv, json
        folder = Path(__file__).resolve().parent
        csv_file = folder / 'targeting_error_sweep.csv'
        json_file = folder / 'targeting_error_results.json'
        if csv_file.exists() and json_file.exists():
            with csv_file.open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            data = json.loads(json_file.read_text())
            baseline = next(x for x in rows if x['stage'] == 'baseline_validation')
            self.assertAlmostEqual(float(baseline['trajectory_success_rate']),
                                   data['baseline_validation']['trajectory_success_rate'])
            self.assertEqual(int(baseline['trials']), 12000)
            self.assertEqual(len([x for x in rows if x['stage'] == 'coarse']), 17)
            self.assertEqual(data['maximum_tolerable_sigma_mm'], None)

    def test_wilson_uncertainty(self):
        """Confidence intervals contain observed rate and narrow with n."""
        low, high = wilson_interval(950, 1000)
        self.assertLess(low, .95)
        self.assertGreater(high, .95)
        low2, high2 = wilson_interval(9500, 10000)
        self.assertLess(high2 - low2, high - low)


if __name__ == "__main__":
    unittest.main(verbosity=2)
