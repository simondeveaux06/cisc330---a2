"""Targeting Error Simulation — CISC/CMPE 330 bone biopsy assignment.

Run from the directory containing registration.py, geometry.py, simulation.py:
    python targeting_error_simulation.py

All distances and per-coordinate standard deviations are in millimetres.
The existing M_TOOL values have ALREADY been converted from cm to mm.

Primary outcome: shortest distance from the estimated needle trajectory to
its displaced target (the assignment's definition of compound targeting error).
Secondary outcome: distance from the needle tip after exactly 100 mm of travel
to that target (the PDF also uses fixed-depth needle-tip wording).
"""

import csv
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np

from geometry import point_line_distance
from registration import M_TOOL, P_TOOL, V_TOOL, register_tool
from simulation import marker_sigma

SEED = 3302026
INSERTION_DEPTH_MM = 100.0
ERROR_LIMIT_MM = 3.0
SUCCESS_TARGET = 0.95
MARKER_SIGMA_MM = marker_sigma(radius=1.0, probability=0.95)

# Coarse sweep first, then a finer sweep around the crossing.
COARSE_STEP_MM = 0.25
COARSE_MAX_MM = 4.0
FINE_STEP_MM = 0.025
COARSE_TRIALS = 2000
FINE_TRIALS = 6000
FINAL_TRIALS = 12000
OUTPUT_CSV = Path('targeting_error_sweep.csv')
OUTPUT_JSON = Path('targeting_error_results.json')
OUTPUT_SUCCESS_PLOT = Path('targeting_success_probability.png')
OUTPUT_ERROR_PLOT = Path('targeting_error_vs_sigma.png')


def wilson_interval(successes, trials, confidence=0.95):
    """Binomial confidence interval for the observed success proportion."""
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    margin = z * np.sqrt(p * (1 - p) / trials + z * z / (4 * trials**2)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def make_trial_data(n, rng):
    """Draw independent noise for each marker and for target motion.

    Reuse the same standard-normal target draws across different sigma values
    within a sweep. This 'common random numbers' approach reduces artificial
    differences between adjacent sigma settings, without coupling marker and
    target errors within any one trial.
    """
    marker_noise = rng.normal(0.0, MARKER_SIGMA_MM, size=(n, 3, 3))
    target_standard_normals = rng.normal(0.0, 1.0, size=(n, 3))
    return marker_noise, target_standard_normals


def estimated_needle_poses(marker_noise):
    """Re-register all three independently perturbed markers for EVERY trial."""
    n = len(marker_noise)
    tips = np.empty((n, 3))
    directions = np.empty((n, 3))
    for i in range(n):
        noisy_markers = M_TOOL + marker_noise[i]  # M_TOOL is already in mm
        tips[i], directions[i], _ = register_tool(
            M_TOOL, P_TOOL, V_TOOL, noisy_markers
        )
    return tips, directions


def evaluate_sigma(sigma_target, tips, directions, target_normals):
    """Evaluate a sigma using identical registration samples for each level."""
    # With F_model == F_CT, the ideal target is 100 mm down the true axis.
    ideal_target = P_TOOL + INSERTION_DEPTH_MM * V_TOOL
    displaced_targets = ideal_target + sigma_target * target_normals

    # Explicitly reuse the group's point-to-line distance function.
    trajectory_errors = np.array([
        point_line_distance(target, tip, direction)
        for target, tip, direction in zip(displaced_targets, tips, directions)
    ])

    # Secondary interpretation: where is the tip after precisely 100 mm?
    inserted_tips = tips + INSERTION_DEPTH_MM * directions
    tip_errors = np.linalg.norm(displaced_targets - inserted_tips, axis=1)

    n = len(trajectory_errors)
    trajectory_successes = int(np.count_nonzero(trajectory_errors <= ERROR_LIMIT_MM))
    tip_successes = int(np.count_nonzero(tip_errors <= ERROR_LIMIT_MM))
    low, high = wilson_interval(trajectory_successes, n)
    tip_low, tip_high = wilson_interval(tip_successes, n)
    return {
        'sigma_target_mm': float(sigma_target), 'trials': n,
        'trajectory_success_rate': trajectory_successes / n,
        'trajectory_ci_low': low, 'trajectory_ci_high': high,
        'trajectory_p95_mm': float(np.quantile(trajectory_errors, 0.95)),
        'trajectory_mean_mm': float(trajectory_errors.mean()),
        'tip_success_rate': tip_successes / n,
        'tip_ci_low': tip_low, 'tip_ci_high': tip_high,
        'tip_p95_mm': float(np.quantile(tip_errors, 0.95)),
    }


def run_sweep(sigmas, n, rng):
    """Prepare independent trials, register each once, then test each sigma."""
    marker_noise, target_normals = make_trial_data(n, rng)
    tips, directions = estimated_needle_poses(marker_noise)
    return [evaluate_sigma(s, tips, directions, target_normals) for s in sigmas]


def crossing_bracket(rows):
    """Find first sampled level failing the 95% success criterion."""
    for index, row in enumerate(rows):
        if row['trajectory_success_rate'] < SUCCESS_TARGET:
            if index == 0:
                raise RuntimeError('Even sigma=0 fails; no feasible target noise found.')
            return rows[index - 1]['sigma_target_mm'], row['sigma_target_mm']
    raise RuntimeError('No crossing found; increase COARSE_MAX_MM.')


def print_row(row):
    print(f"  sigma={row['sigma_target_mm']:6.3f} mm | "
          f"trajectory success={100 * row['trajectory_success_rate']:6.2f}% "
          f"(95% Wilson CI {100 * row['trajectory_ci_low']:5.2f}–"
          f"{100 * row['trajectory_ci_high']:5.2f}%) | "
          f"p95={row['trajectory_p95_mm']:6.3f} mm | "
          f"tip success={100 * row['tip_success_rate']:6.2f}%")



def save_outputs(tagged_rows, status, boundary, baseline_validation=None):
    """Export all results and recreate both labelled figures on every run."""
    with OUTPUT_CSV.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(tagged_rows[0]))
        writer.writeheader()
        writer.writerows(tagged_rows)

    coarse = [r for r in tagged_rows if r['stage'] == 'coarse']
    payload = {
        'simulation': 'CISC 330 Targeting Error Simulation',
        'seed': SEED, 'length_unit': 'mm',
        'marker_noise_per_coordinate_sigma_mm': float(MARKER_SIGMA_MM),
        'ideal_target_mm': (P_TOOL + INSERTION_DEPTH_MM * V_TOOL).tolist(),
        'insertion_depth_mm': INSERTION_DEPTH_MM,
        'success_threshold_mm': ERROR_LIMIT_MM,
        'required_success_probability': SUCCESS_TARGET,
        'primary_metric': 'shortest distance to infinite estimated needle trajectory',
        'secondary_metric': 'distance to needle tip after 100 mm insertion',
        'coarse_sweep': {'min_sigma_mm': 0.0, 'max_sigma_mm': COARSE_MAX_MM,
                         'step_mm': COARSE_STEP_MM, 'trials_per_level': COARSE_TRIALS,
                         'levels': len(coarse)},
        'fine_sweep_resolution_mm': FINE_STEP_MM,
        'fine_sweep_executed': any(r['stage'] == 'fine' for r in tagged_rows),
        'maximum_tolerable_sigma_mm': boundary.get('maximum_tolerable_sigma_mm'),
        'maximum_status': status, 'boundary': boundary,
        'baseline_validation': baseline_validation,
        'coarse_rows': coarse,
        'all_rows': tagged_rows,
        'uncertainty_method': '95% Wilson interval for binomial success probability',
        'note': 'Shared random draws across sweep levels; independent marker and target draws within each trial.'
    }
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x = np.array([r['sigma_target_mm'] for r in coarse])
    success = np.array([r['trajectory_success_rate'] for r in coarse])
    lo = np.array([r['trajectory_ci_low'] for r in coarse])
    hi = np.array([r['trajectory_ci_high'] for r in coarse])
    tip_success = np.array([r['tip_success_rate'] for r in coarse])
    p95 = np.array([r['trajectory_p95_mm'] for r in coarse])
    mean = np.array([r['trajectory_mean_mm'] for r in coarse])
    tip_p95 = np.array([r['tip_p95_mm'] for r in coarse])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(x, 100 * success, 'o-', label='Trajectory success')
    ax.fill_between(x, 100 * lo, 100 * hi, alpha=0.2,
                    label='95% Wilson confidence interval')
    ax.plot(x, 100 * tip_success, 's--', label='Fixed-depth tip success')
    ax.axhline(100 * SUCCESS_TARGET, linestyle=':', color='firebrick',
               label='Required 95% success')
    if baseline_validation is not None:
        ax.scatter([0], [100 * baseline_validation['trajectory_success_rate']],
                   marker='D', s=70, label='Independent baseline (12,000 trials)', zorder=5)
    ax.set(xlabel='Target-displacement sigma per coordinate (mm)',
           ylabel='Trials within 3 mm (%)',
           title='Targeting success versus target displacement', ylim=(0, 103))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_SUCCESS_PLOT, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(x, p95, 'o-', label='95th-percentile trajectory error')
    ax.plot(x, mean, 'o-', label='Mean trajectory error')
    ax.plot(x, tip_p95, 's--', label='95th-percentile fixed-depth tip error')
    ax.axhline(ERROR_LIMIT_MM, linestyle=':', color='firebrick',
               label='3 mm clinical limit')
    if baseline_validation is not None:
        ax.scatter([0], [baseline_validation['trajectory_p95_mm']],
                   marker='D', s=70, label='Independent baseline p95', zorder=5)
    ax.set(xlabel='Target-displacement sigma per coordinate (mm)',
           ylabel='Error (mm)', title='Targeting error versus target displacement')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_ERROR_PLOT, dpi=180)
    plt.close(fig)
    print(f'Automatically saved: {OUTPUT_CSV}, {OUTPUT_JSON}, '
          f'{OUTPUT_SUCCESS_PLOT}, {OUTPUT_ERROR_PLOT}')


def main():
    # Verify the ideal scene before adding any errors.
    ideal_target = P_TOOL + INSERTION_DEPTH_MM * V_TOOL
    p0, v0, _ = register_tool(M_TOOL, P_TOOL, V_TOOL, M_TOOL)
    assert np.isclose(point_line_distance(ideal_target, p0, v0), 0.0, atol=1e-10)
    assert np.allclose(ideal_target, [0, -100, 0])

    print('Targeting Error Simulation (all distances in mm)')
    print(f'Marker sigma per coordinate: {MARKER_SIGMA_MM:.6f} mm')
    print(f'Ideal target: {ideal_target}; threshold: {ERROR_LIMIT_MM:.1f} mm')
    print(f'Fixed seed: {SEED}; target success: {100 * SUCCESS_TARGET:.1f}%')

    rng = np.random.default_rng(SEED)
    coarse_sigmas = np.arange(0, COARSE_MAX_MM + COARSE_STEP_MM / 2,
                              COARSE_STEP_MM)
    coarse_rows = run_sweep(coarse_sigmas, COARSE_TRIALS, rng)
    # If even zero target motion fails, increasing sigma cannot establish a
    # positive tolerance. Report that outcome instead of inventing a crossing.
    baseline_failed = coarse_rows[0]['trajectory_success_rate'] < SUCCESS_TARGET
    if not baseline_failed:
        coarse_lo, coarse_hi = crossing_bracket(coarse_rows)
    print(f'\nCoarse sweep ({COARSE_TRIALS} trials/level; '
          f'{COARSE_STEP_MM} mm resolution):')
    for row in coarse_rows:
        print_row(row)
    if baseline_failed:
        # Check the zero-displacement result with more independent trials.
        baseline_check = run_sweep([0.0], FINAL_TRIALS, rng)[0]
        print('\nIndependent baseline check (target sigma = 0):')
        print_row(baseline_check)
        tagged = ([{'stage': 'coarse', **r} for r in coarse_rows]
                  + [{'stage': 'baseline_validation', **baseline_check}])
        save_outputs(tagged,
                     'No feasible nonnegative target-displacement sigma: baseline at sigma=0 fails.',
                     {'last_passing_sigma_mm': None,
                      'first_failing_sigma_mm': 0.0,
                      'maximum_tolerable_sigma_mm': None,
                      'fine_sweep_executed': False,
                      'reason': 'Baseline fails; no passing/failing boundary to refine.'},
                     baseline_validation=baseline_check)
        print('\nNO FEASIBLE TARGET SIGMA: even zero target displacement '
              'fails the 95% trajectory-success requirement under this model.')
        print('No 95% crossing to refine; a finer sweep would not fix the baseline.')
        print(f'Coarse sweep resolution: {COARSE_STEP_MM:.3f} mm; '
              f'baseline validation: {FINAL_TRIALS} trials.')
        print(f'Results saved to {OUTPUT_CSV}')
        return
    print(f'Coarse crossing bracket: [{coarse_lo:.3f}, {coarse_hi:.3f}] mm')
    # Expand the fine window beyond the coarse bracket by one coarse step on
    # either side to protect against Monte Carlo fluctuations.
    fine_min = max(0.0, coarse_lo - COARSE_STEP_MM)
    fine_max = coarse_hi + COARSE_STEP_MM
    fine_sigmas = np.round(np.arange(fine_min, fine_max + FINE_STEP_MM / 2,
                                     FINE_STEP_MM), 6)
    fine_rows = run_sweep(fine_sigmas, FINE_TRIALS, rng)
    fine_lo, fine_hi = crossing_bracket(fine_rows)
    print(f'\nFine sweep ({FINE_TRIALS} trials/level; '
          f'{FINE_STEP_MM} mm resolution):')
    for row in fine_rows:
        print_row(row)
    print(f'Fine crossing bracket: [{fine_lo:.3f}, {fine_hi:.3f}] mm')

    # A new, larger independent sample checks the two boundary candidates.
    final_rows = run_sweep([fine_lo, fine_hi], FINAL_TRIALS, rng)
    print(f'\nIndependent boundary check ({FINAL_TRIALS} trials/level):')
    for row in final_rows:
        print_row(row)

    # Export full coarse/fine tables plus boundary validation for the report.
    tagged = ([{'stage': 'coarse', **r} for r in coarse_rows]
              + [{'stage': 'fine', **r} for r in fine_rows]
              + [{'stage': 'validation', **r} for r in final_rows])
    validated_lo, validated_hi = final_rows
    if (validated_lo['trajectory_success_rate'] >= SUCCESS_TARGET
            and validated_hi['trajectory_success_rate'] < SUCCESS_TARGET):
        print(f'\nEstimated largest passing sampled sigma: {fine_lo:.3f} mm')
        print(f'Sweep resolution: {FINE_STEP_MM:.3f} mm '
              f'(boundary lies between adjacent tested levels).')
    else:
        print('\nWARNING: Independent validation did not reproduce the exact '
              'passing/failing bracket. Increase trials or refine again; '
              'do not report a definitive maximum from this run.')

    if validated_lo['trajectory_ci_low'] >= SUCCESS_TARGET:
        print('At the lower bracket level, the 95% Wilson lower bound also '
              'meets the 95% success requirement.')
    else:
        print('At the lower bracket level, the Wilson interval overlaps '
              '95%; the sampled pass is not a 95%-confidence guarantee.')
    valid_bracket = (validated_lo['trajectory_success_rate'] >= SUCCESS_TARGET
                     and validated_hi['trajectory_success_rate'] < SUCCESS_TARGET)
    save_outputs(tagged,
                 'Validated sampled crossing' if valid_bracket else
                 'Independent boundary validation inconclusive',
                 {'last_passing_sigma_mm': fine_lo if valid_bracket else None,
                  'first_failing_sigma_mm': fine_hi if valid_bracket else None,
                  'maximum_tolerable_sigma_mm': fine_lo if valid_bracket else None,
                  'fine_sweep_executed': True,
                  'resolution_mm': FINE_STEP_MM,
                  'validation_conclusive': bool(valid_bracket)})
    print(f'Full results saved to {OUTPUT_CSV}')
    print('Primary metric: trajectory distance. Secondary: fixed-depth tip distance.')


if __name__ == '__main__':
    main()
