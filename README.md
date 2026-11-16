# cisc330---a2
# CISC/CMPE 330 Assignment 2

CT-guided bone biopsy: tool calibration, tool-to-CT registration, and biopsy
planning for a tracked needle with three markers.

## Reproduce the computational results

Run all commands from the repository root, which contains this README. Every
module and test file sits in the root, and the files import each other by
name, so the file names must stay lowercase exactly as listed below.
No data files, image viewer, or manual measurements are needed. Generated
files under `results/` are created by the commands below.

### 1. Set up Python

Python 3.9 or newer and NumPy are the only requirements. The results below
were checked with Python 3.13 and NumPy 2.5.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, use `python` instead of `python3` to create the environment, then
activate it with `.venv\Scripts\activate.bat` in Command Prompt or
`.venv\Scripts\Activate.ps1` in PowerShell. All subsequent commands use the
activated environment's `python`.

### 2. Run the tests

Each test file is a standalone script. Run each one with `python`:

```bash
python test_registration.py
python test_biopsy_plan.py
python test_calibration.py
```

| Command | Expected final line | What it checks |
| --- | --- | --- |
| `python test_registration.py` | `3/3 tests passed` | Registration: pure translation, pure rotation, rotation + translation |
| `python test_biopsy_plan.py` | `3/3 tests passed` | Biopsy plan: needle intersects, touches, and misses the tumor |
| `python test_calibration.py` | `6/6 tests passed` | Calibration: exact recovery, frame changes, rejected motions, noisy recovery |

Each test prints its predicted and computed values side by side, followed by
`PASS` or `FAIL`. `test_calibration.py` ends with a summary table and exits
with code 1 if any test fails.

Use these scripts rather than `pytest` to grade the results. `pytest` will
collect the 12 test functions, but the registration and biopsy-plan tests
report `PASS`/`FAIL` by printing rather than with `assert`, so `pytest` would
count them as passed even if a value were wrong.

### 3. Regenerate the calibration simulation

```bash
python simulation.py
```

This takes about 10 seconds. It calibrates the tool from exact simulated
observations, then repeats the calibration over 500 trials with fresh Gaussian
marker noise (seed 330). It prints the results and writes the full numbers to
`results/calibration_results.json`.

## How the tasks relate

The tasks form one pipeline. Each task's output is the next task's input:

```
Tool Calibration          tip and axis in the marker frame
  calibration.py            -> calibration_to_model() -> Ptool, vtool (model frame)
        |
        v
Registration              tool model frame -> CT frame
  registration.py           register_tool() -> Pct, vct
        |
        v
Biopsy Plan               needle line vs spherical tumor
  biopsy_plan.py            plan_biopsy() -> targeting error, core length, insertion depth
```

`test_calibration.py` Test 1 runs this chain end to end. It calibrates the
tool, converts the result to the model frame, places the model in a known CT
pose, and checks that `register_tool` recovers the expected tip and axis in CT.

### Module structure

| File | Role | Uses |
| --- | --- | --- |
| `geometry.py` | Shared math: 4×4 homogeneous transforms, frame from three points, rigid registration, rotations, point-to-line distance, line–sphere intersection | NumPy only |
| `fitting.py` | Least-squares plane fit for the axis calibration | NumPy only |
| `registration.py` | Registration task; also defines the ground-truth tool `M_TOOL`, `P_TOOL`, `V_TOOL` | `geometry.py` |
| `biopsy_plan.py` | Biopsy Plan task | `geometry.py` |
| `calibration.py` | Tool Calibration task: pivot calibration for the tip, spin calibration for the axis | `geometry.py`, `fitting.py` |
| `simulation.py` | Generates pivot and spin motions, sets the noise level, runs the noisy trials | `calibration.py`, `geometry.py`, `registration.py` |

`geometry.py` and `fitting.py` know nothing about tools, tumors, or CT. They
take points and vectors and return numbers, which is why every task can reuse
them.

### Which test file runs which task

| Test file | Task | Modules exercised |
| --- | --- | --- |
| `test_registration.py` | Registration | `registration.py`, `geometry.py` |
| `test_biopsy_plan.py` | Biopsy Plan | `biopsy_plan.py`, `geometry.py` |
| `test_calibration.py` | Tool Calibration, and its hand-off to Registration | `calibration.py`, `fitting.py`, `simulation.py`, `registration.py`, `geometry.py` |

## Expected results

### Registration (`test_registration.py`)

The Figure 2 tool is placed in three known CT poses. The CT markers, tip, and
axis were worked out by hand and typed into the test as numbers.

| Test | Pose | Predicted Pct (mm) | Predicted vct | Result |
| --- | --- | --- | --- | --- |
| 1 | Translation by (10, 20, 30) | (10, 20, 30) | (0, −1, 0) | PASS |
| 2 | 90° about z | (0, 0, 0) | (1, 0, 0) | PASS |
| 3 | 90° about x, then translation by (100, −50, 200) | (100, −50, 200) | (0, 0, −1) | PASS |

All tip, axis, and marker RMS errors print as 0. The matrix error for Tests 2
and 3 prints as about 1e-16, which is floating-point rounding in `cos(90°)`.

### Biopsy Plan (`test_biopsy_plan.py`)

The tumor is a unit sphere at the origin. The needle starts at (x0, 0, −10)
and points along (0, 0, 1), so only the sideways offset x0 changes.

| Test | x0 | Status | Targeting error | Core length | Insertion depth |
| --- | ---: | --- | ---: | ---: | ---: |
| 1 | 0.6 | intersects | 0.6 | 1.6 | 10.8 |
| 2 | 1 | touches | 1 | 0 | 10 (contact point) |
| 3 | 2 | misses | 2 | 0 | None |

### Tool Calibration (`test_calibration.py`)

| Test | Checks | Expected result |
| --- | --- | --- |
| 1 | Exact recovery, then hand-off to registration | Tip (−40, −200, 0) mm and axis (0, −1, 0) in the marker frame; pivot (30, −20, 80) mm. Errors near 1e-13 or smaller |
| 2 | Tracker frame rotated and translated | Same marker-frame tip and axis; the pivot moves with the tracker frame |
| 3 | Markers cyclically reordered | Recovers the tip and axis in the new marker frame and converts back to the original model values |
| 4 | Model tip away from the origin, at (8, −3, 11) | Recovers the shifted tip and the unchanged axis |
| 5 | Motions that cannot determine the calibration | Stationary poses, spin-only pivot data, collinear plane points, and an undefined axis sign are all rejected |
| 6 | One noisy calibration | About 95% of marker errors are below 1 mm; tip error below 2 mm and axis error below 3° (observed: about 0.28 mm and 0.03°) |

### Calibration simulation (`python simulation.py`)

Exact data recovers the tip and axis with errors of about 8e-14 mm and 2e-15°.
With noise, over 500 trials:

| Error metric | Mean | SD | Median | P95 | RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tip position (mm) | 0.2788 | 0.1167 | 0.2643 | 0.4911 | 0.3022 |
| Axis direction (deg) | 0.0562 | 0.0296 | 0.0530 | 0.1097 | 0.0635 |

Marker noise check: coordinate σ = 0.35772 mm; 95.02% of 360,000 simulated
marker errors fall below 1 mm.

| Output under `results/` | Contents |
| --- | --- |
| `calibration_results.json` | Settings, exact-data estimates and errors, noisy error statistics, mean estimates, and the observed marker-noise coverage |

The last few digits may vary by machine and NumPy version.

## Conventions used to interpret the results

- All lengths are in mm. Figure 2 is in cm, so `registration.py` multiplies its
  marker coordinates by 10 (`CM_TO_MM`).
- Ground-truth tool, model frame: M1 = (20, 180, 0), M2 = (80, 180, 0),
  M3 = (20, 240, 0), tip P = (0, 0, 0), axis v = (0, −1, 0).
- Transforms are 4×4 homogeneous matrices with the column vector on the right:
  `p_to = F @ p_from`. A transform built from rotation R and translation t is
  `F = T R`, so it rotates first and then translates.
- The marker frame follows Math Primer Part 1: origin at the marker centroid,
  `i = normalize(M2 − M1)`, `k = normalize(i × (M3 − M1))`, `j = k × i`.
  Registration follows Part 3: `F_ct←tool = F_hv · F_he⁻¹`.
- Markers must be given in the same order, M1, M2, M3, in every frame.
- Points are padded with w = 1, so they are rotated and translated. Directions
  are padded with w = 0, so they are only rotated, then renormalized.
- Biopsy plan: the needle is the ray `X(s) = Pct + s·vct` with s ≥ 0, where s
  is the distance in mm from the current tip.
  - Targeting error is the distance from the tumor center to the infinite
    needle axis.
  - A trajectory counts as touching when |d − R| ≤ 1e-9 mm. It reports a core
    length of 0 and the depth to the contact point.
  - A tumor behind the needle is reported as `misses`.
  - If the tip is already inside the tumor, the core starts at the tip.
- Calibration: the tip comes from pivot motions (tip fixed, tool tilted about
  several axes); the axis comes from spin motions (shaft fixed, tool rotated
  about it). The axis sign is chosen to point toward the calibrated tip.
- Noise model: isotropic Gaussian error on each marker coordinate. σ is chosen
  so that 95% of 3-D marker errors are below 1.0 mm, using the chi distribution
  with three degrees of freedom: σ ≈ 0.358 mm.
