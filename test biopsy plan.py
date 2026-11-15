"""
test_biopsy_plan.py — ground-truth tests for the biopsy plan.

Simplest general set-up (as the assignment hints): the tumour is a unit
sphere at the origin, C = (0, 0, 0), R = 1. The needle starts at
P = (x0, 0, -10) and points straight up, v = (0, 0, 1). Then:
  * the closest approach to the centre is at s_c = 10
  * the targeting error is simply d = |x0|
  * if d < 1, the half-chord is h = sqrt(1 - d^2), core = 2h, depth = 10 + h
All expected values below were worked out by hand from these formulas.

Run:  python test_biopsy_plan.py
"""

import numpy as np
from biopsy_plan import plan_biopsy

TOL = 1e-9
C_UNIT, R_UNIT = np.array([0.0, 0.0, 0.0]), 1.0
UP = np.array([0.0, 0.0, 1.0])


def check(name, c, r, p, v, expected):
    out = plan_biopsy(c, r, p, v)
    ok = out["status"] == expected["status"]
    for key in ("targeting_error", "core_length"):
        ok &= abs(out[key] - expected[key]) < TOL
    if expected["insertion_depth"] is None:
        ok &= out["insertion_depth"] is None
    else:
        ok &= out["insertion_depth"] is not None and \
              abs(out["insertion_depth"] - expected["insertion_depth"]) < TOL

    fmt = lambda x: "None" if x is None else f"{x:.6g}"
    print(f"\n=== {name} ===")
    print(f"  {'':18}{'expected':>12}{'computed':>12}")
    print(f"  {'status':18}{expected['status']:>12}{out['status']:>12}")
    for key in ("targeting_error", "core_length", "insertion_depth"):
        print(f"  {key:18}{fmt(expected[key]):>12}{fmt(out[key]):>12}")
    print("  PASS" if ok else "  FAIL")
    return ok


# --------------------------------------------------------------------------
# Test 1: INTERSECTS.  x0 = 0.6
#   d = 0.6, h = sqrt(1 - 0.36) = 0.8   (a 0.6-0.8-1 right triangle)
#   enters at s = 10 - 0.8 = 9.2, exits at s = 10 + 0.8 = 10.8
#   core = 1.6, depth = 10.8
# --------------------------------------------------------------------------
def test_intersect():
    return check("Test 1: intersects (x0 = 0.6)",
                 C_UNIT, R_UNIT, [0.6, 0, -10], UP,
                 {"status": "intersects", "targeting_error": 0.6,
                  "core_length": 1.6, "insertion_depth": 10.8})


# --------------------------------------------------------------------------
# Test 2: TOUCHES (tangent).  x0 = 1
#   d = 1 = R, h = 0: one contact point at (1, 0, 0), s = 10
#   core = 0, depth to the contact point = 10
# --------------------------------------------------------------------------
def test_tangent():
    return check("Test 2: touches / tangent (x0 = 1)",
                 C_UNIT, R_UNIT, [1, 0, -10], UP,
                 {"status": "touches", "targeting_error": 1.0,
                  "core_length": 0.0, "insertion_depth": 10.0})


# --------------------------------------------------------------------------
# Test 3: MISSES.  x0 = 2
#   d = 2 > R: no intersection. core = 0, no insertion depth.
# --------------------------------------------------------------------------
def test_miss():
    return check("Test 3: misses (x0 = 2)",
                 C_UNIT, R_UNIT, [2, 0, -10], UP,
                 {"status": "misses", "targeting_error": 2.0,
                  "core_length": 0.0, "insertion_depth": None})


if __name__ == "__main__":
    results = [test_intersect(), test_tangent(), test_miss()]
    print(f"\n{sum(results)}/{len(results)} tests passed")
